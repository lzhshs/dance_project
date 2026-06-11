"""IK-based retargeting v2.

Keys:
  * Lower body, hips, knees, waist: same Euler decomposition as v1.
  * Shoulders/elbows: computed via IK from SMPL world-frame arm directions.

Algorithm per frame:
  1. Run SMPL FK -> joint world positions (Y-up), then rotate into MuJoCo Z-up.
  2. Set lower-body qpos as before and mj_forward(model, data).
  3. For each arm, read G1 shoulder body's world orientation from data.
     Transform the target upper-arm direction into shoulder local frame.
     Solve 2-DoF pointing IK: pitch (Y), roll (X), yaw=0.
  4. Elbow angle = angle between upper-arm and forearm vectors.
"""
import os
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from retarget_smpl_to_g1 import G1_XML, load_motion
from smpl_fk import fk

OUT_DIR = "/Users/lucy_lzh/dance_project/videos"
os.makedirs(OUT_DIR, exist_ok=True)

# Y-up SMPL -> Z-up MuJoCo
F = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
R_F = R.from_matrix(F)

ARM_SMOOTH_WINDOW = 9
ARM_MAX_STEP = {
    "shoulder": 0.16,
    "elbow": 0.12,
    "wrist": 0.10,
}


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average for smooth joint trajectories."""
    if window <= 1 or values.shape[0] < 3:
        return values
    window = min(window, values.shape[0] if values.shape[0] % 2 else values.shape[0] - 1)
    if window <= 1:
        return values
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")


def _rate_limit(values: np.ndarray, max_step: float) -> np.ndarray:
    """Limit per-frame joint angle changes to reduce physically impossible snaps."""
    out = values.copy()
    for i in range(1, len(out)):
        delta = np.clip(out[i] - out[i - 1], -max_step, max_step)
        out[i] = out[i - 1] + delta
    return out


def _physicalize_arm_joints(qpos: np.ndarray, adr: dict) -> None:
    """Dampen, smooth, and rate-limit G1 upper-body joints in-place.

    EDGE/SMPL can produce expressive arm motion that is plausible for a human
    mesh but too abrupt for the smaller G1 arms. This pass keeps the dance
    gesture while removing high-frequency snaps and excessive wrist twisting.
    """
    for side in ("left", "right"):
        shoulder_joints = [
            f"{side}_shoulder_pitch_joint",
            f"{side}_shoulder_roll_joint",
            f"{side}_shoulder_yaw_joint",
        ]
        elbow_joints = [f"{side}_elbow_joint"]
        wrist_joints = [
            f"{side}_wrist_roll_joint",
            f"{side}_wrist_pitch_joint",
            f"{side}_wrist_yaw_joint",
        ]

        for joint_name in shoulder_joints:
            a = adr[joint_name]
            qpos[:, a] *= 0.82
            qpos[:, a] = _rate_limit(_moving_average(qpos[:, a], ARM_SMOOTH_WINDOW),
                                     ARM_MAX_STEP["shoulder"])

        for joint_name in elbow_joints:
            a = adr[joint_name]
            qpos[:, a] *= 0.75
            qpos[:, a] = _rate_limit(_moving_average(qpos[:, a], ARM_SMOOTH_WINDOW),
                                     ARM_MAX_STEP["elbow"])

        for joint_name in wrist_joints:
            a = adr[joint_name]
            qpos[:, a] *= 0.20
            qpos[:, a] = _rate_limit(_moving_average(qpos[:, a], ARM_SMOOTH_WINDOW),
                                     ARM_MAX_STEP["wrist"])


def solve_arm_ik(target_local: np.ndarray):
    """Given a unit target direction in shoulder local frame, compute
    (pitch, roll) such that R_y(pitch) @ R_x(roll) @ [0, 0, -1] = target.

    Returns (pitch, roll). Yaw is set to 0 (rotation around arm axis is unset).
    """
    d = target_local / (np.linalg.norm(target_local) + 1e-9)
    dx, dy, dz = d
    # R_x(r) @ [0,0,-1] = [0, sin(r), -cos(r)]
    # R_y(p) @ [0, sin(r), -cos(r)] = [-sin(p)*cos(r), sin(r), -cos(p)*cos(r)]
    roll = np.arcsin(np.clip(dy, -1.0, 1.0))
    c = np.cos(roll)
    if abs(c) < 1e-6:
        return 0.0, float(roll)
    pitch = np.arctan2(-dx / c, -dz / c)
    return float(pitch), float(roll)


def build_qpos_trajectory(model, data, poses, trans):
    """Return (T, nq) qpos trajectory for G1 using v1 for lower body + IK for arms."""
    from retarget_smpl_to_g1 import smpl_to_g1_qpos
    T = poses.shape[0]

    # Start from v1 retargeting (lower body correct, arms approximate).
    qpos = smpl_to_g1_qpos(model, poses, trans)

    # SMPL joint world positions + world rotations (Y-up).
    joints_smpl, R_smpl = fk(poses, trans, return_R=True)
    # Rotate into MuJoCo Z-up frame for positions.
    joints_mj = joints_smpl @ F.T  # broadcast

    # Joint name -> qpos adr
    adr = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i):
           int(model.jnt_qposadr[i]) for i in range(model.njnt)}

    # Body name -> body id (for reading world rotation of shoulder_link)
    def bid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

    L_SHOULDER_BODY = bid("left_shoulder_pitch_link")
    R_SHOULDER_BODY = bid("right_shoulder_pitch_link")
    TORSO_BODY = bid("torso_link")
    assert L_SHOULDER_BODY >= 0 and R_SHOULDER_BODY >= 0 and TORSO_BODY >= 0

    # Zero out arm joints so xmat[shoulder_body] returns the REST orientation
    # of the shoulder frame (rotation to which the pitch/roll/yaw chain is
    # relative). Otherwise the v1 (wrong) shoulder angles bias the IK target.
    arm_joints = [f"{s}_{j}_joint" for s in ("left_shoulder", "right_shoulder")
                  for j in ("pitch", "roll", "yaw")] + \
                 ["left_elbow_joint", "right_elbow_joint"]

    # SMPL indices: 16 L_shoulder, 18 L_elbow, 20 L_wrist; 17/19/21 for right.
    # SMPL "torso" rotation: spine3 (joint 9) — the common ancestor of both
    # shoulders via the collars. We express arm directions in this frame so
    # the result is torso-relative and independent of root/world orientation.
    for t in range(T):
        data.qpos[:] = qpos[t]
        for jn in arm_joints:
            data.qpos[adr[jn]] = 0.0
        mujoco.mj_forward(model, data)

        # SMPL torso rotation (Y-up world). Arm positions are in Y-up too.
        R_torso_smpl = R_smpl[t, 9]  # spine3 world rotation in SMPL Y-up
        # G1 torso world rotation (Z-up). This is our target frame for the arm.
        R_torso_g1 = data.xmat[TORSO_BODY].reshape(3, 3)

        for (ls, le, lw, prefix, body_id) in [
            (16, 18, 20, "left_shoulder", L_SHOULDER_BODY),
            (17, 19, 21, "right_shoulder", R_SHOULDER_BODY),
        ]:
            # SMPL arm vector in Y-up world.
            shoulder = joints_smpl[t, ls]
            elbow    = joints_smpl[t, le]
            wrist    = joints_smpl[t, lw]

            upper = elbow - shoulder
            fore  = wrist - elbow
            upper_len = np.linalg.norm(upper)
            if upper_len < 1e-6:
                continue
            # Torso-local arm direction from SMPL.
            # SMPL frame:  X=subject left, Y=up,      Z=forward
            # G1   frame:  X=forward,      Y=left,    Z=up
            # Map SMPL (x,y,z) -> G1 (z, x, y).
            arm_torso_smpl = R_torso_smpl.T @ (upper / upper_len)
            sx, sy, sz = arm_torso_smpl
            arm_torso_g1 = np.array([sz, sx, sy])
            # Back to world (Z-up) via G1 torso rotation.
            upper_dir_world = R_torso_g1 @ arm_torso_g1

            # Shoulder body's world rotation matrix (3x3, row-major in xmat)
            R_shoulder_world = data.xmat[body_id].reshape(3, 3)
            target_local = R_shoulder_world.T @ upper_dir_world
            pitch, roll = solve_arm_ik(target_local)
            qpos[t, adr[f"{prefix}_pitch_joint"]] = pitch
            qpos[t, adr[f"{prefix}_roll_joint"]]  = roll
            qpos[t, adr[f"{prefix}_yaw_joint"]]   = 0.0

            # Elbow angle = angle between upper and forearm directions.
            if np.linalg.norm(fore) > 1e-6:
                cosang = np.dot(upper, fore) / (upper_len * np.linalg.norm(fore))
                cosang = np.clip(cosang, -1.0, 1.0)
                elbow_flex = np.arccos(cosang)
                elbow_name = "left_elbow_joint" if prefix == "left_shoulder" else "right_elbow_joint"
                qpos[t, adr[elbow_name]] = elbow_flex

    _physicalize_arm_joints(qpos, adr)

    # Re-clamp joint limits (pitch/roll may exceed).
    for i in range(1, model.njnt):
        lo, hi = model.jnt_range[i]
        if lo < hi:
            a = int(model.jnt_qposadr[i])
            qpos[:, a] = np.clip(qpos[:, a], lo, hi)
    return qpos


def main():
    # Positional: pkl. Optional: --fps <N> (default 60 for AIST++, 30 for EDGE).
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    fps = 60
    for a in sys.argv[1:]:
        if a.startswith("--fps="):
            fps = int(a.split("=", 1)[1])
    pkl = args[0] if args else \
        "/Users/lucy_lzh/dance_project/aist_data/motions/gPO_sFM_cAll_d10_mPO1_ch02.pkl"
    # Heuristic: EDGE pkls live in generated_motions/ and are 30 fps.
    if "generated_motions" in pkl and "--fps=" not in " ".join(sys.argv):
        fps = 30
    stem = os.path.splitext(os.path.basename(pkl))[0]
    out_mp4 = os.path.join(OUT_DIR, f"{stem}_v2.mp4")
    step = max(1, int(round(fps / 30)))

    model = mujoco.MjModel.from_xml_path(G1_XML)
    data  = mujoco.MjData(model)

    print(f"Loading {pkl}")
    poses, trans = load_motion(pkl)
    print(f"  frames={poses.shape[0]}")

    print("Retargeting v2 (lower body Euler + arm IK) ...")
    qpos_traj = build_qpos_trajectory(model, data, poses, trans)

    print("Rendering ...")
    renderer = mujoco.Renderer(model, height=480, width=640)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.distance = 4.0
    cam.azimuth = 135
    cam.elevation = -15

    frames = []
    for k in range(qpos_traj.shape[0] // step):
        data.qpos[:] = qpos_traj[k * step]
        mujoco.mj_forward(model, data)
        cam.lookat[:] = data.qpos[0:3]
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())

    print(f"Writing {out_mp4}")
    imageio.mimwrite(out_mp4, frames, fps=30, quality=8)
    print("Done.")


if __name__ == "__main__":
    main()
