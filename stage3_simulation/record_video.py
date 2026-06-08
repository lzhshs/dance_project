"""
录制 MuJoCo 仿真视频

使用方法:
    conda activate sim
    python record_video.py --model ../outputs/smpl_humanoid.xml \
                           --motion ../outputs/retargeted_motions/strategy_a/ballet_qpos.npy \
                           --output ../outputs/videos/ballet_demo.mp4
"""
import argparse
import os
import sys
import numpy as np

try:
    import mujoco
except ImportError:
    print("ERROR: pip install mujoco")
    exit(1)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mujoco_utils import load_model_safe


def record_kinematic(model_path, trajectory_path, output_path,
                     width=1280, height=720, fps=30):
    """
    运动学回放录制（直接设 qpos）
    适用于验证动作正确性
    """
    model = load_model_safe(model_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)

    trajectory = np.load(trajectory_path)
    frames = []

    print(f"Recording {len(trajectory)} frames at {fps} fps...")

    for t in range(len(trajectory)):
        nq = min(len(trajectory[t]), model.nq)
        data.qpos[:nq] = trajectory[t][:nq]
        mujoco.mj_forward(model, data)

        renderer.update_scene(data)
        frame = renderer.render()
        frames.append(frame.copy())

        if (t + 1) % 100 == 0:
            print(f"  Frame {t+1}/{len(trajectory)}")

    # 保存视频
    save_video(frames, output_path, fps)


def record_simulation(model_path, sim_qpos_path, output_path,
                      width=1280, height=720, fps=30):
    """
    录制物理仿真结果（从已保存的 sim qpos 回放）
    """
    model = load_model_safe(model_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)

    sim_qpos = np.load(sim_qpos_path)
    frames = []

    print(f"Recording {len(sim_qpos)} frames...")

    for t in range(len(sim_qpos)):
        data.qpos[:] = sim_qpos[t]
        mujoco.mj_forward(model, data)

        renderer.update_scene(data)
        frame = renderer.render()
        frames.append(frame.copy())

    save_video(frames, output_path, fps)


def save_video(frames, output_path, fps):
    """保存帧序列为视频"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # 优先用 opencv（不需要系统 ffmpeg）
    try:
        import cv2
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"Video saved: {output_path} ({len(frames)} frames, {len(frames)/fps:.1f}s)")
        return
    except ImportError:
        pass

    # fallback: mediapy
    try:
        import mediapy
        mediapy.write_video(output_path, frames, fps=fps)
        print(f"Video saved: {output_path} ({len(frames)} frames, {len(frames)/fps:.1f}s)")
        return
    except Exception:
        pass

    # 最后的 fallback: 保存为图片序列
    img_dir = output_path.replace(".mp4", "_frames")
    os.makedirs(img_dir, exist_ok=True)
    from PIL import Image
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(os.path.join(img_dir, f"frame_{i:05d}.png"))
    print(f"Saved {len(frames)} frames to {img_dir}")


def record_side_by_side(model_path, kinematic_qpos, physics_qpos, output_path,
                        width=640, height=720, fps=30):
    """
    并排对比录制: 左边运动学回放，右边物理仿真
    """
    model = load_model_safe(model_path)
    data_kin = mujoco.MjData(model)
    data_phys = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)

    kin_traj = np.load(kinematic_qpos)
    phys_traj = np.load(physics_qpos)
    T = min(len(kin_traj), len(phys_traj))

    frames = []
    for t in range(T):
        # 运动学
        nq = min(len(kin_traj[t]), model.nq)
        data_kin.qpos[:nq] = kin_traj[t][:nq]
        mujoco.mj_forward(model, data_kin)
        renderer.update_scene(data_kin)
        frame_kin = renderer.render().copy()

        # 物理
        data_phys.qpos[:] = phys_traj[t]
        mujoco.mj_forward(model, data_phys)
        renderer.update_scene(data_phys)
        frame_phys = renderer.render().copy()

        # 拼接
        combined = np.concatenate([frame_kin, frame_phys], axis=1)
        frames.append(combined)

    save_video(frames, output_path, fps)
    print(f"Side-by-side video: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Record MuJoCo simulation video")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--motion", type=str, required=True, help="qpos trajectory .npy")
    parser.add_argument("--output", type=str, default="../outputs/videos/demo.mp4")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--sim_qpos", type=str, default=None,
                        help="Physics sim qpos for side-by-side comparison")
    args = parser.parse_args()

    if args.sim_qpos:
        record_side_by_side(
            args.model, args.motion, args.sim_qpos,
            args.output, args.width // 2, args.height, args.fps,
        )
    else:
        record_kinematic(
            args.model, args.motion, args.output,
            args.width, args.height, args.fps,
        )


if __name__ == "__main__":
    main()
