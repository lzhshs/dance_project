"""Stage-2 MuJoCo rollout search for G1 dance references.

This is a lightweight shooting-style optimizer: it tries conservative variants
of an offline-optimized qpos reference, simulates each variant without pinning
the floating base, scores stability/tracking/smoothness, and saves the best
reference for rendering.
"""
from __future__ import annotations

import argparse
import json
import os

import mujoco
import numpy as np

from motion_constraints import actuator_qpos_addresses, joint_addresses, moving_average, rate_limit
from retarget_smpl_to_g1 import G1_XML


OUT_DIR = "/Users/lucy_lzh/dance_project/optimized_motions"
FALL_PELVIS_Z = 0.35


def main():
    parser = argparse.ArgumentParser(description="Rollout-search a G1 optimized reference")
    parser.add_argument("ref", help="Input optimized .npz")
    parser.add_argument("--out", help="Output rollout-optimized .npz")
    parser.add_argument("--max-seconds", type=float, default=10.0)
    args = parser.parse_args()

    source = np.load(args.ref, allow_pickle=True)
    qpos_ref = np.asarray(source["qpos_ref"], dtype=np.float64)
    fps = float(source["fps"][0]) if "fps" in source else 30.0

    model = mujoco.MjModel.from_xml_path(G1_XML)
    candidates = build_candidates(model, qpos_ref)

    print(f"Testing {len(candidates)} rollout candidates ...")
    results = []
    for name, candidate in candidates:
        stats = simulate_candidate(model, candidate, fps, args.max_seconds)
        score = score_stats(stats, candidate, fps)
        results.append((score, name, candidate, stats))
        print(f"{name:18s} score={score:7.3f} stability={stats['stability_rate']*100:5.1f}% "
              f"mean_err={stats['mean_joint_error']:.3f} pelvis_min={stats['pelvis_z_min']:.2f}")

    score, name, best, stats = min(results, key=lambda item: item[0])
    stem = os.path.splitext(os.path.basename(args.ref))[0].replace("_optimized", "")
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = args.out or os.path.join(OUT_DIR, f"{stem}_rollout_optimized.npz")
    np.savez_compressed(
        out_path,
        qpos_ref=best.astype(np.float32),
        fps=np.array([fps], dtype=np.float32),
        source=np.array([args.ref]),
        candidate_name=np.array([name]),
        rollout_stats=json.dumps(stats),
    )
    print(f"\nSelected: {name} score={score:.3f}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"Wrote {out_path}")


def build_candidates(model, qpos_ref: np.ndarray) -> list[tuple[str, np.ndarray]]:
    specs = [
        ("offline", 1.00, 1.00, 1.00, 1.00),
        ("balanced_90", 0.90, 0.90, 0.85, 0.90),
        ("balanced_75", 0.75, 0.78, 0.70, 0.78),
        ("lower_safe", 0.62, 0.72, 0.55, 0.70),
        ("upper_safe", 0.85, 0.55, 0.50, 0.80),
        ("very_safe", 0.55, 0.50, 0.35, 0.58),
    ]
    return [(name, make_candidate(model, qpos_ref, leg, arm, waist, root))
            for name, leg, arm, waist, root in specs]


def make_candidate(model, qpos_ref: np.ndarray, leg_gain: float, arm_gain: float,
                   waist_gain: float, root_gain: float) -> np.ndarray:
    qpos = qpos_ref.copy()
    adr = joint_addresses(model)

    qpos[:, 0] = qpos[0, 0] + root_gain * (qpos[:, 0] - qpos[0, 0])
    qpos[:, 1] = qpos[0, 1] + root_gain * (qpos[:, 1] - qpos[0, 1])
    qpos[:, 2] = qpos[:, 2].mean() + 0.35 * (qpos[:, 2] - qpos[:, 2].mean())

    for joint_name, address in adr.items():
        if address < 7:
            continue
        if any(token in joint_name for token in ["hip", "knee", "ankle"]):
            gain, max_step = leg_gain, 0.060
        elif any(token in joint_name for token in ["shoulder", "elbow", "wrist"]):
            gain, max_step = arm_gain, 0.050
        elif "waist" in joint_name:
            gain, max_step = waist_gain, 0.040
        else:
            gain, max_step = 0.75, 0.050
        qpos[:, address] = rate_limit(moving_average(qpos[:, address] * gain, 11), max_step)

    for joint_id in range(1, model.njnt):
        low, high = model.jnt_range[joint_id]
        if low < high:
            address = int(model.jnt_qposadr[joint_id])
            qpos[:, address] = np.clip(qpos[:, address], low, high)
    return qpos


def simulate_candidate(model, qpos_ref: np.ndarray, fps: float, max_seconds: float) -> dict[str, float | None]:
    data = mujoco.MjData(model)
    ctrl_to_qadr = actuator_qpos_addresses(model)
    duration = min(qpos_ref.shape[0] / fps, max_seconds)
    dt = model.opt.timestep
    n_steps = int(duration / dt)

    data.qpos[:] = qpos_ref[0]
    data.qpos[2] += 0.03
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    err_sum = 0.0
    err_count = 0
    pelvis_z_min = float("inf")
    fall_time = None
    for step in range(n_steps):
        t_sec = step * dt
        q_ref = interp_qpos(qpos_ref, t_sec, fps)
        data.ctrl[:] = q_ref[ctrl_to_qadr]
        mujoco.mj_step(model, data)

        err_sum += float(np.abs(data.qpos[ctrl_to_qadr] - q_ref[ctrl_to_qadr]).mean())
        err_count += 1
        pelvis_z = float(data.qpos[2])
        pelvis_z_min = min(pelvis_z_min, pelvis_z)
        if fall_time is None and pelvis_z < FALL_PELVIS_Z:
            fall_time = t_sec
            break

    stability_rate = 1.0 if fall_time is None else fall_time / max(duration, 1e-9)
    return {
        "duration": duration,
        "fall_time": fall_time,
        "stability_rate": stability_rate,
        "mean_joint_error": err_sum / max(err_count, 1),
        "pelvis_z_min": pelvis_z_min,
    }


def interp_qpos(qpos_ref: np.ndarray, t_sec: float, fps: float) -> np.ndarray:
    idx_f = t_sec * fps
    i0 = min(int(np.floor(idx_f)), qpos_ref.shape[0] - 1)
    i1 = min(i0 + 1, qpos_ref.shape[0] - 1)
    alpha = idx_f - i0
    return (1.0 - alpha) * qpos_ref[i0] + alpha * qpos_ref[i1]


def score_stats(stats: dict[str, float | None], qpos_ref: np.ndarray, fps: float) -> float:
    actuated_smooth = float(np.abs(np.diff(qpos_ref[:, 7:], axis=0)).mean()) * fps
    fall_penalty = 100.0 * (1.0 - float(stats["stability_rate"]))
    track_penalty = 8.0 * float(stats["mean_joint_error"])
    height_penalty = max(0.0, 0.60 - float(stats["pelvis_z_min"])) * 6.0
    smooth_penalty = 0.15 * actuated_smooth
    motion_reward = -0.05 * float(np.std(qpos_ref[:, 7:]))
    return fall_penalty + track_penalty + height_penalty + smooth_penalty + motion_reward


if __name__ == "__main__":
    main()
