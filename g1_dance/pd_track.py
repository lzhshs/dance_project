"""Physics-based PD tracking of a kinematic/optimized qpos trajectory.

The G1 scene.xml defines 29 position actuators with built-in affine-bias PD
(Kp=500, Kd from biasprm[2]). We drive them by setting data.ctrl to the
reference joint targets from the retargeted trajectory, then let MuJoCo step
with full rigid-body dynamics + ground contacts.

Outputs:
  - artifacts/videos/<stem>_pd.mp4  : rendered physics simulation
  - prints tracking stats (mean/max joint error, fall rate, falls at time)

Usage: python -m g1_dance.pd_track <motion.pkl|optimized.npz> [--pin-root]
       --pin-root  Overwrite floating-base qpos/qvel each step from the
                   reference trajectory, isolating joint tracking from the
                   balance problem.
"""
import argparse
import os
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np

from g1_dance.retarget_smpl_to_g1 import G1_XML, load_motion
from g1_dance.retarget_v2 import build_qpos_trajectory

from g1_dance.project_paths import AIST_MOTIONS_DIR, VIDEOS_DIR

OUT_DIR = str(VIDEOS_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

MOTION_FPS = 60.0       # AIST++ is 60 fps; EDGE outputs 30 fps (override via --fps)
RENDER_FPS = 30
FALL_PELVIS_Z = 0.35    # below this => considered fallen


def interp_qpos(qpos_traj: np.ndarray, t_sec: float) -> np.ndarray:
    """Linearly interpolate reference qpos at time t_sec from 60 fps trajectory."""
    idx_f = t_sec * MOTION_FPS
    i0 = int(np.floor(idx_f))
    i1 = min(i0 + 1, qpos_traj.shape[0] - 1)
    i0 = min(i0, qpos_traj.shape[0] - 1)
    a = idx_f - i0
    return (1.0 - a) * qpos_traj[i0] + a * qpos_traj[i1]


def main():
    global MOTION_FPS
    parser = argparse.ArgumentParser(description="PD-track a G1 reference trajectory")
    parser.add_argument("input", nargs="?",
                        default=str(AIST_MOTIONS_DIR / "gPO_sFM_cAll_d10_mPO1_ch02.pkl"),
                        help="SMPL .pkl motion or optimized .npz reference")
    parser.add_argument("--ref", help="Optimized .npz reference; overrides input")
    parser.add_argument("--fps", type=float, default=0, help="Reference FPS")
    parser.add_argument("--pin-root", action="store_true",
                        help="Overwrite floating-base state from reference each step")
    parsed = parser.parse_args()

    ref_path = parsed.ref or (parsed.input if parsed.input.endswith(".npz") else None)
    source_path = ref_path or parsed.input
    stem = os.path.splitext(os.path.basename(source_path))[0]
    pin_root = parsed.pin_root
    suffix = "_pd_pinned" if pin_root else "_pd"
    out_mp4 = os.path.join(OUT_DIR, f"{stem}{suffix}.mp4")

    model = mujoco.MjModel.from_xml_path(G1_XML)
    data = mujoco.MjData(model)

    if ref_path:
        print(f"Loading optimized reference {ref_path}")
        ref = np.load(ref_path, allow_pickle=True)
        qpos_traj = np.asarray(ref["qpos_ref"], dtype=np.float64)
        MOTION_FPS = float(parsed.fps or (ref["fps"][0] if "fps" in ref else MOTION_FPS))
    else:
        pkl = parsed.input
        if parsed.fps > 0:
            MOTION_FPS = parsed.fps
        elif "generated_motions" in pkl:
            MOTION_FPS = 30.0
        print(f"Loading {pkl}")
        poses, trans = load_motion(pkl)

        print("Retargeting (v2 IK) ...")
        qpos_traj = build_qpos_trajectory(model, data, poses, trans)
    T_frames = qpos_traj.shape[0]
    duration = T_frames / MOTION_FPS
    print(f"  {T_frames} frames, {duration:.1f}s")

    # Build actuator -> qpos index map
    ctrl_to_qadr = np.zeros(model.nu, dtype=np.int64)
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]
        ctrl_to_qadr[i] = model.jnt_qposadr[jid]

    # Initialize robot to first kinematic frame so it starts in-pose, standing.
    # Lift slightly to avoid initial ground penetration, then let physics settle.
    data.qpos[:] = qpos_traj[0]
    data.qpos[2] += 0.02
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    # Renderer
    W, H = 640, 480
    renderer = mujoco.Renderer(model, height=H, width=W)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.distance = 4.0
    cam.azimuth = 135
    cam.elevation = -15

    # Simulation loop
    dt = model.opt.timestep  # 0.002
    n_steps = int(duration / dt)
    render_every = int(round(1.0 / (RENDER_FPS * dt)))  # 500/30 ≈ 17

    # Stats
    joint_err_sum = np.zeros(model.nu)
    joint_err_cnt = 0
    pelvis_z_log = []
    fall_time = None
    frames = []

    print(f"Simulating {n_steps} steps ({duration:.1f}s) @ dt={dt*1000:.1f}ms ...")
    for step in range(n_steps):
        t_sec = step * dt
        q_ref = interp_qpos(qpos_traj, t_sec)

        # Set position actuator targets to reference joint angles.
        for a_i in range(model.nu):
            data.ctrl[a_i] = q_ref[ctrl_to_qadr[a_i]]

        mujoco.mj_step(model, data)

        if pin_root:
            # Overwrite floating-base DoFs with reference values so the robot
            # cannot fall; joints still follow PD dynamics.
            data.qpos[0:7] = q_ref[0:7]
            data.qvel[0:6] = 0.0

        # Tracking error
        for a_i in range(model.nu):
            err = data.qpos[ctrl_to_qadr[a_i]] - q_ref[ctrl_to_qadr[a_i]]
            joint_err_sum[a_i] += abs(err)
        joint_err_cnt += 1

        pelvis_z = float(data.qpos[2])
        pelvis_z_log.append(pelvis_z)
        if fall_time is None and pelvis_z < FALL_PELVIS_Z:
            fall_time = t_sec

        if step % render_every == 0:
            cam.lookat[:] = data.qpos[0:3]
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())

    mean_err = joint_err_sum / max(joint_err_cnt, 1)
    print("\n=== Tracking stats ===")
    print(f"Mean |joint error| (rad): {mean_err.mean():.3f}")
    print(f"Max  |joint error| (rad): {mean_err.max():.3f}  "
          f"({mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(np.argmax(mean_err)))})")
    pelvis_z_log = np.array(pelvis_z_log)
    print(f"Pelvis z: min={pelvis_z_log.min():.2f}  "
          f"mean={pelvis_z_log.mean():.2f}  max={pelvis_z_log.max():.2f}")
    if fall_time is not None:
        print(f"FALL at t={fall_time:.2f}s (pelvis z < {FALL_PELVIS_Z})")
        stability_rate = fall_time / duration
    else:
        print("Did not fall.")
        stability_rate = 1.0
    print(f"Physical stability rate: {stability_rate*100:.1f}%")

    print(f"Writing {out_mp4}")
    imageio.mimwrite(out_mp4, frames, fps=RENDER_FPS, quality=8)
    print("Done.")


if __name__ == "__main__":
    main()
