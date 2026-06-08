"""
端到端运行脚本: 一键执行整个 pipeline (Stage 2→3→Eval)
1. 生成新的 MuJoCo XML 模型 (position actuators)
2. 重新 retarget 所有动作
3. 物理仿真 (kinematic root)
4. 录制视频
5. 运行评估

Usage:
    conda activate sim
    cd <project>
    python run_all.py
"""
import os
import sys
import numpy as np

PROJECT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT, "stage2_retarget"))
sys.path.insert(0, os.path.join(PROJECT, "stage3_simulation"))
sys.path.insert(0, os.path.join(PROJECT, "evaluation"))

from mujoco_utils import load_model_safe

# ---- 路径 ----
MODEL_PATH = os.path.join(PROJECT, "outputs", "smpl_humanoid.xml")
MOTION_DIR = os.path.join(PROJECT, "outputs", "generated_motions")
RETARGET_DIR = os.path.join(PROJECT, "outputs", "retargeted_motions", "strategy_a")
SIM_DIR = os.path.join(PROJECT, "outputs", "sim_results")
VIDEO_DIR = os.path.join(PROJECT, "outputs", "videos")
MUSIC_DIR = os.path.join(PROJECT, "music")
EVAL_OUTPUT = os.path.join(PROJECT, "outputs", "evaluation_results.json")

GENRES = ["ballet", "hiphop", "house"]


def step1_generate_model():
    """生成新的 MuJoCo humanoid XML (position actuators)"""
    print("\n" + "="*60)
    print("  STEP 1: Generate MuJoCo humanoid model")
    print("="*60)
    from retarget_smplsim import generate_minimal_humanoid
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    generate_minimal_humanoid(MODEL_PATH)

    # 验证模型
    import mujoco
    model = load_model_safe(MODEL_PATH)
    print(f"  Model: nq={model.nq}, nv={model.nv}, nu={model.nu}, njnt={model.njnt}")
    print(f"  Actuator type: position (built-in PD)")


def step2_retarget():
    """重新 retarget 所有动作"""
    print("\n" + "="*60)
    print("  STEP 2: Retarget SMPL motions → MuJoCo qpos")
    print("="*60)
    from retarget_smplsim import convert_motion_to_qpos
    os.makedirs(RETARGET_DIR, exist_ok=True)

    for genre in GENRES:
        motion_path = os.path.join(MOTION_DIR, f"{genre}_motion.npy")
        if not os.path.exists(motion_path):
            print(f"  SKIP: {motion_path} not found")
            continue
        output_path = os.path.join(RETARGET_DIR, f"{genre}_qpos.npy")
        print(f"\n  Retargeting: {genre}")
        convert_motion_to_qpos(motion_path, MODEL_PATH, output_path)


def step3_simulate():
    """物理仿真 (kinematic root mode)"""
    print("\n" + "="*60)
    print("  STEP 3: Physics simulation (kinematic root)")
    print("="*60)
    from pd_controller import SimulationRunner
    os.makedirs(SIM_DIR, exist_ok=True)

    runner = SimulationRunner(MODEL_PATH)
    print(f"  Model: nq={runner.model.nq}, nv={runner.model.nv}, nu={runner.model.nu}")

    for genre in GENRES:
        qpos_path = os.path.join(RETARGET_DIR, f"{genre}_qpos.npy")
        if not os.path.exists(qpos_path):
            print(f"  SKIP: {qpos_path} not found")
            continue

        print(f"\n  Simulating: {genre}")
        trajectory = np.load(qpos_path)
        print(f"    Trajectory: {trajectory.shape}")

        results = runner.run_physics(trajectory, render=False, kinematic_root=True)

        # 保存结果
        for key, val in results.items():
            np.save(os.path.join(SIM_DIR, f"{genre}_{key}.npy"), val)

        # 统计
        stable = int(np.sum(~results["fell"]))
        total = len(results["fell"])
        print(f"    PSR: {stable}/{total} = {100*stable/total:.1f}%")


def step4_record_videos():
    """录制视频"""
    print("\n" + "="*60)
    print("  STEP 4: Record videos")
    print("="*60)
    from record_video import save_video
    import mujoco
    os.makedirs(VIDEO_DIR, exist_ok=True)

    model = load_model_safe(MODEL_PATH)

    for genre in GENRES:
        qpos_path = os.path.join(RETARGET_DIR, f"{genre}_qpos.npy")
        sim_qpos_path = os.path.join(SIM_DIR, f"{genre}_qpos.npy")

        if not os.path.exists(qpos_path):
            print(f"  SKIP: {genre} (no retargeted motion)")
            continue

        kin_traj = np.load(qpos_path)
        has_sim = os.path.exists(sim_qpos_path)

        # --- Kinematic video ---
        print(f"\n  Recording {genre} kinematic...")
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=720, width=1280)
        frames = []
        for t in range(len(kin_traj)):
            nq = min(len(kin_traj[t]), model.nq)
            data.qpos[:nq] = kin_traj[t][:nq]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data)
            frames.append(renderer.render().copy())
        save_video(frames, os.path.join(VIDEO_DIR, f"{genre}_kinematic.mp4"), 30)

        # --- Physics video ---
        if has_sim:
            print(f"  Recording {genre} physics...")
            sim_traj = np.load(sim_qpos_path)
            frames_phys = []
            for t in range(len(sim_traj)):
                nq = min(len(sim_traj[t]), model.nq)
                data.qpos[:nq] = sim_traj[t][:nq]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data)
                frames_phys.append(renderer.render().copy())
            save_video(frames_phys, os.path.join(VIDEO_DIR, f"{genre}_physics.mp4"), 30)

            # --- Side-by-side comparison ---
            print(f"  Recording {genre} comparison...")
            data_kin = mujoco.MjData(model)
            data_phys = mujoco.MjData(model)
            renderer_half = mujoco.Renderer(model, height=720, width=640)
            T = min(len(kin_traj), len(sim_traj))
            frames_cmp = []
            for t in range(T):
                nq_k = min(len(kin_traj[t]), model.nq)
                data_kin.qpos[:nq_k] = kin_traj[t][:nq_k]
                mujoco.mj_forward(model, data_kin)
                renderer_half.update_scene(data_kin)
                fk = renderer_half.render().copy()

                nq_p = min(len(sim_traj[t]), model.nq)
                data_phys.qpos[:nq_p] = sim_traj[t][:nq_p]
                mujoco.mj_forward(model, data_phys)
                renderer_half.update_scene(data_phys)
                fp = renderer_half.render().copy()

                frames_cmp.append(np.concatenate([fk, fp], axis=1))
            save_video(frames_cmp, os.path.join(VIDEO_DIR, f"{genre}_comparison.mp4"), 30)


def step5_evaluate():
    """运行评估"""
    print("\n" + "="*60)
    print("  STEP 5: Evaluation")
    print("="*60)
    from beat_alignment import compute_bas
    from stability import compute_psr
    from joint_violation import compute_jlvr
    import json

    results = []

    for genre in GENRES:
        print(f"\n  Evaluating: {genre}")
        r = {"genre": genre}

        # BAS
        wav_path = os.path.join(MUSIC_DIR, f"{genre}.wav")
        motion_path = os.path.join(MOTION_DIR, f"{genre}_motion.npy")
        if os.path.exists(wav_path) and os.path.exists(motion_path):
            try:
                bas = compute_bas(wav_path, motion_path)
                r["beat_alignment_score"] = bas["beat_alignment_score"]
                r["tempo"] = bas["tempo"]
                r["n_music_beats"] = bas["n_music_beats"]
                r["n_motion_beats"] = bas["n_motion_beats"]
                print(f"    BAS = {bas['beat_alignment_score']:.4f}")
            except Exception as e:
                r["beat_alignment_score"] = None
                print(f"    BAS error: {e}")
        else:
            r["beat_alignment_score"] = None

        # PSR
        try:
            psr = compute_psr(SIM_DIR, genre)
            r["physical_stability_rate"] = psr["physical_stability_rate"]
            r["first_fall_time"] = psr["first_fall_time"]
            r["com_height_mean"] = psr["com_height_mean"]
            print(f"    PSR = {psr['physical_stability_rate']:.4f}")
        except FileNotFoundError:
            r["physical_stability_rate"] = None
            print(f"    PSR: no sim data")

        # JLVR
        try:
            jlvr = compute_jlvr(SIM_DIR, genre, MODEL_PATH)
            r["joint_limit_violation_rate"] = jlvr["joint_limit_violation_rate"]
            print(f"    JLVR = {jlvr['joint_limit_violation_rate']:.4f}")
        except FileNotFoundError:
            r["joint_limit_violation_rate"] = None
            print(f"    JLVR: no sim data")

        results.append(r)

    # 打印表格
    print("\n" + "="*70)
    print(f"{'Genre':<12} {'BAS':>8} {'PSR':>8} {'JLVR':>8}")
    print("-"*70)
    for r in results:
        bas = f"{r['beat_alignment_score']:.4f}" if r.get("beat_alignment_score") is not None else "N/A"
        psr = f"{r['physical_stability_rate']:.4f}" if r.get("physical_stability_rate") is not None else "N/A"
        jlvr = f"{r['joint_limit_violation_rate']:.4f}" if r.get("joint_limit_violation_rate") is not None else "N/A"
        print(f"{r['genre']:<12} {bas:>8} {psr:>8} {jlvr:>8}")
    print("="*70)

    # 平均值
    bas_vals = [r["beat_alignment_score"] for r in results if r.get("beat_alignment_score") is not None]
    psr_vals = [r["physical_stability_rate"] for r in results if r.get("physical_stability_rate") is not None]
    jlvr_vals = [r["joint_limit_violation_rate"] for r in results if r.get("joint_limit_violation_rate") is not None]
    if bas_vals:
        print(f"Avg BAS:  {np.mean(bas_vals):.4f}")
    if psr_vals:
        print(f"Avg PSR:  {np.mean(psr_vals):.4f}")
    if jlvr_vals:
        print(f"Avg JLVR: {np.mean(jlvr_vals):.4f}")

    # 保存 JSON
    os.makedirs(os.path.dirname(EVAL_OUTPUT), exist_ok=True)
    with open(EVAL_OUTPUT, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {EVAL_OUTPUT}")


if __name__ == "__main__":
    print("="*60)
    print("  Full Pipeline: Retarget → Simulate → Record → Evaluate")
    print("="*60)

    step1_generate_model()
    step2_retarget()
    step3_simulate()
    step4_record_videos()
    step5_evaluate()

    print("\n" + "="*60)
    print("  ALL DONE!")
    print("="*60)
    print(f"  Videos:  {VIDEO_DIR}")
    print(f"  Results: {EVAL_OUTPUT}")
