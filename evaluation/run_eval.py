"""
综合评估脚本
跨风格、跨策略对比评估

使用方法:
    conda activate sim
    python run_eval.py --music_dir ../music \
                       --motion_dir ../outputs/generated_motions \
                       --sim_dir ../outputs/sim_results \
                       --model ../outputs/smpl_humanoid.xml \
                       --output ../outputs/evaluation_results.json

输出: JSON 报告 + 终端打印对比表格
"""
import argparse
import json
import os
import numpy as np

from beat_alignment import compute_bas
from stability import compute_psr
from joint_violation import compute_jlvr


def evaluate_single(genre, music_dir, motion_dir, sim_dir, model_path, fps=30):
    """评估单个音乐风格"""
    result = {"genre": genre}

    wav_path = os.path.join(music_dir, f"{genre}.wav")
    motion_path = os.path.join(motion_dir, f"{genre}_motion.npy")

    # 1. Beat Alignment Score
    if os.path.exists(wav_path) and os.path.exists(motion_path):
        try:
            bas_result = compute_bas(wav_path, motion_path, fps=fps)
            result["beat_alignment_score"] = bas_result["beat_alignment_score"]
            result["tempo"] = bas_result["tempo"]
            result["n_music_beats"] = bas_result["n_music_beats"]
            result["n_motion_beats"] = bas_result["n_motion_beats"]
        except Exception as e:
            print(f"  BAS error: {e}")
            result["beat_alignment_score"] = None
    else:
        result["beat_alignment_score"] = None
        if not os.path.exists(wav_path):
            print(f"  Missing: {wav_path}")
        if not os.path.exists(motion_path):
            print(f"  Missing: {motion_path}")

    # 2. Physical Stability Rate
    try:
        psr_result = compute_psr(sim_dir, genre, fps=fps)
        result["physical_stability_rate"] = psr_result["physical_stability_rate"]
        result["first_fall_time"] = psr_result["first_fall_time"]
        result["com_height_mean"] = psr_result["com_height_mean"]
    except FileNotFoundError:
        result["physical_stability_rate"] = None
        print(f"  No simulation results for {genre}")

    # 3. Joint Limit Violation Rate
    if model_path and os.path.exists(model_path):
        try:
            jlvr_result = compute_jlvr(sim_dir, genre, model_path)
            result["joint_limit_violation_rate"] = jlvr_result["joint_limit_violation_rate"]
        except FileNotFoundError:
            result["joint_limit_violation_rate"] = None
    else:
        result["joint_limit_violation_rate"] = None

    return result


def print_table(results):
    """打印对比表格"""
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    header = f"{'Genre':<12} {'BAS':>8} {'PSR':>8} {'JLVR':>8} {'Tempo':>8} {'Fall(s)':>8}"
    print(header)
    print("-" * 80)

    for r in results:
        bas = f"{r['beat_alignment_score']:.4f}" if r.get("beat_alignment_score") is not None else "N/A"
        psr = f"{r['physical_stability_rate']:.4f}" if r.get("physical_stability_rate") is not None else "N/A"
        jlvr = f"{r['joint_limit_violation_rate']:.4f}" if r.get("joint_limit_violation_rate") is not None else "N/A"
        tempo = f"{r['tempo']:.1f}" if r.get("tempo") is not None else "N/A"
        fall = f"{r['first_fall_time']:.1f}" if r.get("first_fall_time") is not None and r["first_fall_time"] >= 0 else "Never"

        print(f"{r['genre']:<12} {bas:>8} {psr:>8} {jlvr:>8} {tempo:>8} {fall:>8}")

    print("=" * 80)
    print("BAS = Beat Alignment Score (higher is better)")
    print("PSR = Physical Stability Rate (higher is better)")
    print("JLVR = Joint Limit Violation Rate (lower is better)")
    print()

    # 平均值
    bas_vals = [r["beat_alignment_score"] for r in results if r.get("beat_alignment_score") is not None]
    psr_vals = [r["physical_stability_rate"] for r in results if r.get("physical_stability_rate") is not None]
    jlvr_vals = [r["joint_limit_violation_rate"] for r in results if r.get("joint_limit_violation_rate") is not None]

    if bas_vals:
        print(f"Average BAS:  {np.mean(bas_vals):.4f}")
    if psr_vals:
        print(f"Average PSR:  {np.mean(psr_vals):.4f}")
    if jlvr_vals:
        print(f"Average JLVR: {np.mean(jlvr_vals):.4f}")


def compare_strategies(results_a, results_b):
    """对比 Strategy A vs B"""
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON: A (SMPL) vs B (G1)")
    print("=" * 80)

    header = f"{'Genre':<12} {'PSR-A':>8} {'PSR-B':>8} {'JLVR-A':>8} {'JLVR-B':>8}"
    print(header)
    print("-" * 80)

    for ra, rb in zip(results_a, results_b):
        psr_a = f"{ra['physical_stability_rate']:.4f}" if ra.get("physical_stability_rate") is not None else "N/A"
        psr_b = f"{rb['physical_stability_rate']:.4f}" if rb.get("physical_stability_rate") is not None else "N/A"
        jlvr_a = f"{ra['joint_limit_violation_rate']:.4f}" if ra.get("joint_limit_violation_rate") is not None else "N/A"
        jlvr_b = f"{rb['joint_limit_violation_rate']:.4f}" if rb.get("joint_limit_violation_rate") is not None else "N/A"

        print(f"{ra['genre']:<12} {psr_a:>8} {psr_b:>8} {jlvr_a:>8} {jlvr_b:>8}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Comprehensive evaluation")
    parser.add_argument("--music_dir", type=str, default="../music")
    parser.add_argument("--motion_dir", type=str, default="../outputs/generated_motions")
    parser.add_argument("--sim_dir", type=str, default="../outputs/sim_results")
    parser.add_argument("--model", type=str, default="../outputs/smpl_humanoid.xml")
    parser.add_argument("--output", type=str, default="../outputs/evaluation_results.json")
    parser.add_argument("--genres", type=str, nargs="+", default=["hiphop", "ballet", "house"])
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    print("Starting evaluation...")
    print(f"Genres: {args.genres}")

    results = []
    for genre in args.genres:
        print(f"\nEvaluating: {genre}")
        r = evaluate_single(genre, args.music_dir, args.motion_dir, args.sim_dir, args.model, args.fps)
        results.append(r)

    # 打印结果
    print_table(results)

    # 保存 JSON
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        # 移除 numpy arrays（不可 JSON 序列化）
        clean_results = []
        for r in results:
            clean = {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
            clean_results.append(clean)
        json.dump(clean_results, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
