"""
仿真主脚本 — 串联整个 Stage 3 pipeline

使用方法:
    # 1. 先生成模型（只需一次）
    python run_sim.py --generate_model

    # 2. 运动学回放（验证重定向）
    python run_sim.py --mode kinematic --motion ../outputs/retargeted_motions/strategy_a/ballet_qpos.npy --render

    # 3. 物理仿真（kinematic root模式，默认）
    python run_sim.py --mode physics --motion ../outputs/retargeted_motions/strategy_a/ballet_qpos.npy --render

    # 4. 录制视频
    python run_sim.py --mode record --motion ../outputs/retargeted_motions/strategy_a/ballet_qpos.npy

    # 5. 批量处理所有动作
    python run_sim.py --mode batch
"""
import argparse
import os
import sys
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage2_retarget"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pd_controller import SimulationRunner
from record_video import record_kinematic, record_simulation, record_side_by_side, save_video
from mujoco_utils import load_model_safe


def record_physics_video(model_path, sim_qpos, output_path, width=1280, height=720, fps=30):
    """从物理仿真结果录制视频"""
    import mujoco
    model = load_model_safe(model_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)

    frames = []
    for t in range(len(sim_qpos)):
        nq = min(len(sim_qpos[t]), model.nq)
        data.qpos[:nq] = sim_qpos[t][:nq]
        mujoco.mj_forward(model, data)

        renderer.update_scene(data)
        frame = renderer.render()
        frames.append(frame.copy())

    save_video(frames, output_path, fps)


def batch_process(model_path, motion_dir, output_dir, video_dir):
    """批量处理所有重定向动作"""
    motion_files = [f for f in os.listdir(motion_dir) if f.endswith("_qpos.npy")]

    if not motion_files:
        print(f"No motion files in {motion_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    runner = SimulationRunner(model_path)

    for f in motion_files:
        name = f.replace("_qpos.npy", "")
        print(f"\n{'='*50}")
        print(f"Processing: {name}")
        print(f"{'='*50}")

        trajectory = np.load(os.path.join(motion_dir, f))

        # 1. 运动学回放
        print("\n1. Kinematic playback...")
        kin_results = runner.run_kinematic(trajectory, render=False)

        # 2. 物理仿真 (kinematic root)
        print("\n2. Physics simulation (kinematic root)...")
        phys_results = runner.run_physics(trajectory, render=False, kinematic_root=True)

        # 3. 保存结果
        for key, val in phys_results.items():
            np.save(os.path.join(output_dir, f"{name}_{key}.npy"), val)

        # 4. 录制运动学视频
        print("\n3. Recording kinematic video...")
        record_kinematic(
            model_path,
            os.path.join(motion_dir, f),
            os.path.join(video_dir, f"{name}_kinematic.mp4"),
        )

        # 5. 录制物理仿真视频
        print("4. Recording physics video...")
        record_physics_video(
            model_path,
            phys_results["qpos"],
            os.path.join(video_dir, f"{name}_physics.mp4"),
        )

        # 6. 录制对比视频 — 保存临时文件给 side-by-side
        physics_qpos_path = os.path.join(output_dir, f"{name}_qpos.npy")
        np.save(physics_qpos_path, phys_results["qpos"])
        print("5. Recording comparison video...")
        record_side_by_side(
            model_path,
            os.path.join(motion_dir, f),
            physics_qpos_path,
            os.path.join(video_dir, f"{name}_comparison.mp4"),
        )

    print(f"\n\n{'='*50}")
    print(f"Batch complete! Processed {len(motion_files)} motions.")
    print(f"{'='*50}")
    print(f"  Sim results: {output_dir}")
    print(f"  Videos: {video_dir}")


def main():
    parser = argparse.ArgumentParser(description="Simulation pipeline")
    parser.add_argument("--mode", type=str, default="physics",
                        choices=["kinematic", "physics", "record", "batch", "generate_model"])
    parser.add_argument("--model", type=str, default="../outputs/smpl_humanoid.xml")
    parser.add_argument("--motion", type=str, default=None)
    parser.add_argument("--motion_dir", type=str, default="../outputs/retargeted_motions/strategy_a")
    parser.add_argument("--output_dir", type=str, default="../outputs/sim_results")
    parser.add_argument("--video_dir", type=str, default="../outputs/videos")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--full_physics", action="store_true", help="No kinematic root (harder)")
    args = parser.parse_args()

    if args.mode == "generate_model":
        from retarget_smplsim import generate_minimal_humanoid
        os.makedirs(os.path.dirname(args.model) or ".", exist_ok=True)
        generate_minimal_humanoid(args.model)
        print("Model generated. You can now run simulations.")
        return

    if args.mode == "batch":
        batch_process(args.model, args.motion_dir, args.output_dir, args.video_dir)
        return

    if args.motion is None:
        print("ERROR: --motion required for non-batch mode")
        return

    trajectory = np.load(args.motion)
    print(f"Loaded: {args.motion} ({trajectory.shape})")

    runner = SimulationRunner(args.model)

    if args.mode == "kinematic":
        results = runner.run_kinematic(trajectory, render=args.render)
    elif args.mode == "physics":
        kinematic_root = not args.full_physics
        results = runner.run_physics(trajectory, render=args.render,
                                     kinematic_root=kinematic_root)
    elif args.mode == "record":
        os.makedirs(args.video_dir, exist_ok=True)
        name = os.path.basename(args.motion).replace("_qpos.npy", "").replace(".npy", "")
        record_kinematic(
            args.model, args.motion,
            os.path.join(args.video_dir, f"{name}.mp4"),
        )
        return

    # 保存
    os.makedirs(args.output_dir, exist_ok=True)
    name = os.path.basename(args.motion).replace("_qpos.npy", "").replace(".npy", "")
    for key, val in results.items():
        np.save(os.path.join(args.output_dir, f"{name}_{key}.npy"), val)
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
