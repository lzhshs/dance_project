"""
Retarget an AIST++ SMPL motion to a Unitree G1 robot and render a video.

SMPL: 24 joints, axis-angle rotations (72-dim per frame). Y-up coordinate frame.
G1:   29 hinge DoFs + floating base. Z-up coordinate frame.

Strategy: per-frame Euler decomposition.
  For each SMPL body-joint with a corresponding G1 3-DoF chain (pitch->roll->yaw),
  decompose R_smpl in Euler 'YXZ' order (matching G1 joint axes Y, X, Z) and
  fill the three qpos slots.

Mapping (SMPL idx -> G1 joints):
  0  pelvis     -> free joint (pos + quat)
  1  L_Hip      -> left_hip_{pitch,roll,yaw}
  2  R_Hip      -> right_hip_{pitch,roll,yaw}
  3  Spine1     -> waist_{yaw,roll,pitch} (note G1 order)
  4  L_Knee     -> left_knee (Y only)
  5  R_Knee     -> right_knee
  6  Spine2     -> merged into waist, skipped
  7  L_Ankle    -> left_ankle_{pitch,roll}
  8  R_Ankle    -> right_ankle_{pitch,roll}
  9  Spine3     -> skip
  10 L_Foot     -> skip (toe)
  11 R_Foot     -> skip
  12 Neck       -> skip (G1 has no neck)
  13 L_Collar   -> skip
  14 R_Collar   -> skip
  15 Head       -> skip
  16 L_Shoulder -> left_shoulder_{pitch,roll,yaw}
  17 R_Shoulder -> right_shoulder_{pitch,roll,yaw}
  18 L_Elbow    -> left_elbow (Y only)
  19 R_Elbow    -> right_elbow
  20 L_Wrist    -> left_wrist_{roll,pitch,yaw}  (order differs)
  21 R_Wrist    -> right_wrist_{roll,pitch,yaw}
  22 L_Hand     -> skip
  23 R_Hand     -> skip
"""
import pickle
import sys
import os
import numpy as np
import mujoco
import imageio.v2 as imageio
from scipy.spatial.transform import Rotation as R

from g1_dance.project_paths import AIST_MOTIONS_DIR, G1_XML, VIDEOS_DIR

DEFAULT_PKL = str(AIST_MOTIONS_DIR / "gBR_sBM_cAll_d04_mBR0_ch01.pkl")
OUT_DIR = str(VIDEOS_DIR)
SMPL_PKL = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PKL
OUT_MP4 = os.path.join(OUT_DIR,
                       os.path.splitext(os.path.basename(SMPL_PKL))[0] + ".mp4")
os.makedirs(OUT_DIR, exist_ok=True)

# Frame from SMPL (Y-up) to MuJoCo (Z-up): rotate -90deg about X.
# point_mj = F @ point_smpl
F = np.array([[1, 0, 0],
              [0, 0, -1],
              [0, 1, 0]], dtype=np.float64)
R_F = R.from_matrix(F)

def load_motion(path):
    d = pickle.load(open(path, "rb"))
    poses = d["smpl_poses"].astype(np.float64)          # (T, 72)
    scaling = float(d.get("smpl_scaling", [1.0])[0])
    trans = d["smpl_trans"].astype(np.float64) / scaling
    # EDGE-generated pkls have 30 fps while AIST++ ground truth is 60 fps.
    # Mark via `fps` field if present, else infer by filename convention.
    return poses, trans

def smpl_to_g1_qpos(model, poses, trans):
    """Map SMPL motion to G1 qpos trajectory (T, nq)."""
    T = poses.shape[0]
    qpos = np.tile(np.zeros(model.nq), (T, 1))

    # Compute global z offset so the median pelvis height matches G1's standing
    # pelvis (~0.79m). Using median is robust against brief crouches/jumps.
    pelvis_z_smpl = trans[:, 1]  # SMPL Y is vertical
    z_offset = float(np.median(pelvis_z_smpl)) - 0.79

    # joint-to-qpos-adr helper
    adr = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i):
           int(model.jnt_qposadr[i]) for i in range(model.njnt)}

    for t in range(T):
        aa = poses[t].reshape(24, 3)  # axis-angle per joint
        root_trans_smpl = trans[t]

        # --- Root: translation + orientation ---
        root_pos_mj = F @ root_trans_smpl
        root_pos_mj[2] -= z_offset  # normalize vertical so lowest pelvis ~0.55m

        R_root_smpl = R.from_rotvec(aa[0])
        R_root_mj = R_F * R_root_smpl * R_F.inv()
        quat_xyzw = R_root_mj.as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        qpos[t, 0:3] = root_pos_mj
        qpos[t, 3:7] = quat_wxyz

        # --- Helper: decompose axis-angle to Euler YXZ (pitch, roll, yaw) ---
        def euler_yxz(a):
            return R.from_rotvec(a).as_euler("YXZ", degrees=False)

        # Hips (3-DoF chain pitch->roll->yaw)
        for smpl_i, prefix in [(1, "left_hip"), (2, "right_hip")]:
            p, r, y = euler_yxz(aa[smpl_i])
            qpos[t, adr[f"{prefix}_pitch_joint"]] = p
            qpos[t, adr[f"{prefix}_roll_joint"]]  = r
            qpos[t, adr[f"{prefix}_yaw_joint"]]   = y

        # Waist: G1 has one 3-DoF chain (yaw->roll->pitch, axes Z,X,Y).
        # SMPL has spine1/2/3 (joints 3, 6, 9). Compose spine1+spine2 so the
        # torso is more expressive than spine1 alone, but don't add spine3 or
        # neck (they overflow G1's ±30° waist roll/pitch range).
        R_spine = R.from_rotvec(aa[3]) * R.from_rotvec(aa[6])
        wp, wr, wy = R_spine.as_euler("ZXY", degrees=False)
        qpos[t, adr["waist_yaw_joint"]]   = wp
        qpos[t, adr["waist_roll_joint"]]  = wr
        qpos[t, adr["waist_pitch_joint"]] = wy

        # Knees: 1-DoF Y axis. Humans only flex (never hyperextend) so force
        # positive angle: use the magnitude of the SMPL knee rotation.
        qpos[t, adr["left_knee_joint"]]  = abs(np.linalg.norm(aa[4]))
        qpos[t, adr["right_knee_joint"]] = abs(np.linalg.norm(aa[5]))

        # Ankles: 2-DoF (pitch Y, roll X) — order in chain Y then X → YXZ, take p,r
        lap, lar, _ = euler_yxz(aa[7])
        qpos[t, adr["left_ankle_pitch_joint"]] = lap
        qpos[t, adr["left_ankle_roll_joint"]]  = lar
        rap, rar, _ = euler_yxz(aa[8])
        qpos[t, adr["right_ankle_pitch_joint"]] = rap
        qpos[t, adr["right_ankle_roll_joint"]]  = rar

        # Shoulders (3-DoF chain pitch->roll->yaw).
        # Rest-pose alignment: SMPL T-pose arm points along ±X, G1 default arm
        # points along -Z. Pre-align by rotating -Z to ±X before decomposition.
        #   left:  R_y(-90°) maps -Z to +X
        #   right: R_y(+90°) maps -Z to -X
        R_align_L = R.from_euler("y", -np.pi / 2)
        R_align_R = R.from_euler("y", +np.pi / 2)
        for smpl_i, prefix, R_align in [(16, "left_shoulder",  R_align_L),
                                        (17, "right_shoulder", R_align_R)]:
            R_g1 = R.from_rotvec(aa[smpl_i]) * R_align
            p, r, y = R_g1.as_euler("YXZ", degrees=False)
            qpos[t, adr[f"{prefix}_pitch_joint"]] = p
            qpos[t, adr[f"{prefix}_roll_joint"]]  = r
            qpos[t, adr[f"{prefix}_yaw_joint"]]   = y

        # Elbows: 1-DoF Y. Same flex-only convention as knees.
        qpos[t, adr["left_elbow_joint"]]  = abs(np.linalg.norm(aa[18]))
        qpos[t, adr["right_elbow_joint"]] = abs(np.linalg.norm(aa[19]))

        # Wrists: 3-DoF chain roll->pitch->yaw (X, Y, Z) → Euler XYZ
        for smpl_i, prefix in [(20, "left_wrist"), (21, "right_wrist")]:
            wr_, wp_, wy_ = R.from_rotvec(aa[smpl_i]).as_euler("XYZ", degrees=False)
            qpos[t, adr[f"{prefix}_roll_joint"]]  = wr_
            qpos[t, adr[f"{prefix}_pitch_joint"]] = wp_
            qpos[t, adr[f"{prefix}_yaw_joint"]]   = wy_

    # Clamp to joint limits to avoid MuJoCo complaints
    for i in range(1, model.njnt):  # skip free joint
        lo, hi = model.jnt_range[i]
        if lo < hi:
            a = int(model.jnt_qposadr[i])
            qpos[:, a] = np.clip(qpos[:, a], lo, hi)
    return qpos

def main():
    model = mujoco.MjModel.from_xml_path(G1_XML)
    data  = mujoco.MjData(model)

    print(f"Loading SMPL motion from {SMPL_PKL}")
    poses, trans = load_motion(SMPL_PKL)
    print(f"  frames={poses.shape[0]}, duration={poses.shape[0]/60:.1f}s")

    print("Retargeting to G1 qpos ...")
    qpos_traj = smpl_to_g1_qpos(model, poses, trans)
    print(f"  qpos_traj shape: {qpos_traj.shape}")

    # Render. Use downsampled fps (AIST++ is 60 fps; we render 30).
    STEP = 2
    T_render = qpos_traj.shape[0] // STEP
    print(f"Rendering {T_render} frames @ 30fps ...")

    renderer = mujoco.Renderer(model, height=480, width=640)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.distance = 4.0
    cam.azimuth = 135
    cam.elevation = -15

    frames = []
    for k in range(T_render):
        data.qpos[:] = qpos_traj[k * STEP]
        mujoco.mj_forward(model, data)
        cam.lookat[:] = data.qpos[0:3]  # follow robot
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())

    print(f"Writing {OUT_MP4}")
    imageio.mimwrite(OUT_MP4, frames, fps=30, quality=8)
    print("Done.")

if __name__ == "__main__":
    main()
