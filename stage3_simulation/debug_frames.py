"""Extract a few representative frames from each video for inspection."""
import os
import numpy as np
import mujoco
import cv2

MODEL = "../outputs/smpl_humanoid.xml"
MOTION_DIR = "../outputs/retargeted_motions/strategy_a"
SIM_DIR = "../outputs/sim_results"
OUT_DIR = "../outputs/debug_frames"
os.makedirs(OUT_DIR, exist_ok=True)

model = mujoco.MjModel.from_xml_path(MODEL)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=480, width=640)

for genre in ["ballet", "hiphop", "house"]:
    # Kinematic (target motion)
    traj = np.load(os.path.join(MOTION_DIR, f"{genre}_qpos.npy"))
    T = len(traj)
    sample_frames = [0, T//4, T//2, 3*T//4, T-1]

    for i, frame_idx in enumerate(sample_frames):
        data.qpos[:model.nq] = traj[frame_idx][:model.nq]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data)
        img = renderer.render()
        out_path = os.path.join(OUT_DIR, f"{genre}_kin_t{frame_idx:03d}.png")
        cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # Physics
    phys_path = os.path.join(SIM_DIR, f"{genre}_qpos.npy")
    if os.path.exists(phys_path):
        traj_phys = np.load(phys_path)
        T = len(traj_phys)
        sample_frames = [0, T//4, T//2, 3*T//4, T-1]
        for frame_idx in sample_frames:
            data.qpos[:] = traj_phys[frame_idx]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data)
            img = renderer.render()
            out_path = os.path.join(OUT_DIR, f"{genre}_phys_t{frame_idx:03d}.png")
            cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # Print stats
    print(f"\n{genre}:")
    print(f"  Kinematic qpos range:")
    print(f"    root_pos: {traj[:, :3].min(axis=0)} ~ {traj[:, :3].max(axis=0)}")
    print(f"    root_quat_w: {traj[:, 3].min():.3f} ~ {traj[:, 3].max():.3f}")
    if os.path.exists(phys_path):
        print(f"  Physics qpos range:")
        print(f"    root_pos: {traj_phys[:, :3].min(axis=0)} ~ {traj_phys[:, :3].max(axis=0)}")
        print(f"    com_height mean: {np.load(os.path.join(SIM_DIR, f'{genre}_com_height.npy')).mean():.3f}")

print("\nFrames saved to", OUT_DIR)
