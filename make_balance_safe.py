"""Create a conservative no-pin-safe G1 reference.

This keeps the lower body very close to the Unitree G1 stand keyframe while
retaining a damped version of the upper-body dance. It is the current practical
fallback when full-body dance references cannot stand without root pinning.
"""
from __future__ import annotations

import argparse
import json
import os

import mujoco
import numpy as np

from retarget_smpl_to_g1 import G1_XML
from rollout_optimize import make_candidate, score_stats, simulate_candidate


OUT_DIR = "/Users/lucy_lzh/dance_project/optimized_motions"


def main():
    parser = argparse.ArgumentParser(description="Build a conservative no-pin-safe reference")
    parser.add_argument("ref", help="Input optimized .npz")
    parser.add_argument("--out", help="Output .npz")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(G1_XML)
    source = np.load(args.ref, allow_pickle=True)
    qpos_ref = np.asarray(source["qpos_ref"], dtype=np.float64)
    fps = float(source["fps"][0]) if "fps" in source else 30.0
    configs = [
        ("leg00_arm35_root00", 0.0, 0.35, 0.05, 0.0),
        ("leg05_arm35_root00", 0.05, 0.35, 0.05, 0.0),
        ("leg10_arm45_root05", 0.10, 0.45, 0.08, 0.05),
        ("leg20_arm45_root10", 0.20, 0.45, 0.10, 0.10),
        ("leg30_arm55_root20", 0.30, 0.55, 0.15, 0.20),
    ]

    results = []
    for name, leg_gain, arm_gain, waist_gain, root_gain in configs:
        candidate = make_candidate(model, qpos_ref, leg_gain, arm_gain, waist_gain, root_gain)
        candidate[:, 7:19] *= 0.15 if leg_gain > 0 else 0.0
        candidate[:, 19:22] *= waist_gain
        stats = simulate_candidate(model, candidate, fps, 10.0)
        candidate_score = score_stats(stats, candidate, fps)
        results.append((candidate_score, name, candidate, stats))
        print(f"{name:22s} score={candidate_score:.4f} stability={stats['stability_rate']*100:.1f}% "
              f"err={stats['mean_joint_error']:.4f}")

    score, name, best, stats = min(results, key=lambda item: item[0])
    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.ref))[0].replace("_optimized", "")
    out_path = args.out or os.path.join(OUT_DIR, f"{stem}_balance_safe.npz")
    np.savez_compressed(
        out_path,
        qpos_ref=best.astype(np.float32),
        fps=np.array([fps], dtype=np.float32),
        source=np.array([args.ref]),
        candidate_name=np.array([name]),
        rollout_stats=json.dumps(stats),
    )
    print(f"Selected {name} score={score:.4f}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
