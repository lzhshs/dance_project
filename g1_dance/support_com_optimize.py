"""Support/COM-aware rollout optimizer for G1 dance references.

This is a final-stage optimizer after ``robot_feasibility_optimize.py``.  It
keeps the same body-group gain search, but adds an explicit kinematic support
metric: center-of-mass distance to the active foot-support center.  The selected
reference must still pass unpinned MuJoCo rollout evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from g1_dance.motion_constraints import LEFT_FOOT_BODY, RIGHT_FOOT_BODY, moving_average
from g1_dance.retarget_smpl_to_g1 import G1_XML
from g1_dance.robot_feasibility_optimize import (
    FeasibilityParams,
    build_parameter_set,
    make_feasible_reference,
    motion_expressiveness,
    simulate_feasibility,
)


from g1_dance.project_paths import OPTIMIZED_MOTIONS_DIR

OUT_DIR = str(OPTIMIZED_MOTIONS_DIR)


@dataclass(frozen=True)
class CandidateReport:
    params: dict
    score: float
    expressiveness: float
    stability_rate: float
    fall_time: float | None
    mean_joint_error: float
    pelvis_z_min: float
    support_error_mean: float
    support_error_p95: float
    single_support_fraction: float


def main() -> None:
    parser = argparse.ArgumentParser(description="Support/COM-aware optimize a G1 dance reference")
    parser.add_argument("ref", help="Input .npz, usually artifacts/optimized_motions/pop_optimized.npz")
    parser.add_argument("--out", help="Output .npz path")
    parser.add_argument("--samples", type=int, default=160)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-seconds", type=float, default=10.0)
    parser.add_argument("--min-stability", type=float, default=0.98)
    args = parser.parse_args()

    source = np.load(args.ref, allow_pickle=True)
    qpos_source = np.asarray(source["qpos_ref"], dtype=np.float64)
    fps = float(source["fps"][0]) if "fps" in source else 30.0
    model = mujoco.MjModel.from_xml_path(G1_XML)

    params = build_support_parameter_set(args.samples, args.seed)
    print(f"Testing {len(params)} support/COM-aware candidates ...")

    results = []
    for index, param in enumerate(params, start=1):
        candidate = make_feasible_reference(model, qpos_source, param)
        support = support_metrics(model, candidate, fps)
        rollout = simulate_feasibility(model, candidate, fps, args.max_seconds)
        expr = motion_expressiveness(model, candidate, qpos_source)
        score = support_com_score(rollout, support, expr, args.min_stability)
        report = CandidateReport(
            params=asdict(param),
            score=score,
            expressiveness=expr,
            stability_rate=float(rollout["stability_rate"]),
            fall_time=rollout["fall_time"],
            mean_joint_error=float(rollout["mean_joint_error"]),
            pelvis_z_min=float(rollout["pelvis_z_min"]),
            support_error_mean=float(support["support_error_mean"]),
            support_error_p95=float(support["support_error_p95"]),
            single_support_fraction=float(support["single_support_fraction"]),
        )
        results.append((score, expr, candidate, report))
        print(
            f"{index:03d}/{len(params):03d} score={score:8.3f} "
            f"stable={report.stability_rate*100:5.1f}% expr={expr:5.3f} "
            f"com={report.support_error_mean:.3f}/{report.support_error_p95:.3f} "
            f"leg={param.leg_gain:.2f} arm={param.arm_gain:.2f} waist={param.waist_gain:.2f}"
        )

    score, expr, best, report = min(results, key=lambda item: item[0])
    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.ref))[0].replace("_optimized", "")
    out_path = args.out or os.path.join(OUT_DIR, f"{stem}_support_com.npz")
    final_report = {
        "selected": report.__dict__,
        "min_stability": args.min_stability,
        "source": args.ref,
    }
    np.savez_compressed(
        out_path,
        qpos_ref=best.astype(np.float32),
        qpos_source=qpos_source.astype(np.float32),
        fps=np.array([fps], dtype=np.float32),
        source=np.array([args.ref]),
        support_com_report=json.dumps(final_report, ensure_ascii=False),
    )
    print("\nSelected support/COM-aware reference")
    print(json.dumps(final_report, indent=2, ensure_ascii=False))
    print(f"Wrote {out_path}")


def build_support_parameter_set(samples: int, seed: int) -> list[FeasibilityParams]:
    rng = np.random.default_rng(seed)
    params = [
        FeasibilityParams(0.02, 1.25, 0.03, 0.00, 0.25, 0.03, 0.85),
        FeasibilityParams(0.04, 1.30, 0.05, 0.00, 0.30, 0.03, 0.82),
        FeasibilityParams(0.06, 1.20, 0.06, 0.01, 0.32, 0.04, 0.90),
        FeasibilityParams(0.08, 1.10, 0.08, 0.01, 0.35, 0.04, 0.95),
    ]
    params.extend(build_parameter_set(min(samples, max(20, samples // 2)), seed))
    while len(params) < samples:
        # Bias toward stable lower-body ranges but allow large arms for dance expressiveness.
        leg_gain = float(rng.beta(1.1, 5.2) * 0.32)
        arm_gain = float(0.75 + rng.beta(2.0, 1.5) * 0.75)
        waist_gain = float(rng.beta(1.2, 5.0) * 0.20)
        root_xy_gain = float(rng.beta(1.0, 7.0) * 0.08)
        root_yaw_gain = float(0.15 + rng.beta(1.4, 3.2) * 0.45)
        vertical_gain = float(rng.beta(1.0, 5.5) * 0.08)
        smooth_scale = float(0.72 + rng.beta(1.6, 2.2) * 0.65)
        params.append(
            FeasibilityParams(
                leg_gain=leg_gain,
                arm_gain=arm_gain,
                waist_gain=waist_gain,
                root_xy_gain=root_xy_gain,
                root_yaw_gain=root_yaw_gain,
                vertical_gain=vertical_gain,
                smooth_scale=smooth_scale,
            )
        )
    return params[:samples]


def support_metrics(model: mujoco.MjModel, qpos_ref: np.ndarray, fps: float) -> dict[str, float]:
    data = mujoco.MjData(model)
    left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, LEFT_FOOT_BODY)
    right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, RIGHT_FOOT_BODY)
    if left_id < 0 or right_id < 0:
        raise ValueError("Could not find G1 foot bodies")

    left = np.zeros((len(qpos_ref), 3), dtype=np.float64)
    right = np.zeros_like(left)
    com = np.zeros_like(left)
    for frame, qpos in enumerate(qpos_ref):
        data.qpos[:] = qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        left[frame] = data.xpos[left_id]
        right[frame] = data.xpos[right_id]
        com[frame] = whole_body_com(model, data)

    left_contact = detect_contact_from_positions(left, fps)
    right_contact = detect_contact_from_positions(right, fps)
    both_missing = ~(left_contact | right_contact)
    left_contact[both_missing] = True
    right_contact[both_missing] = True

    support_center = np.zeros((len(qpos_ref), 2), dtype=np.float64)
    for frame in range(len(qpos_ref)):
        feet = []
        if left_contact[frame]:
            feet.append(left[frame, :2])
        if right_contact[frame]:
            feet.append(right[frame, :2])
        support_center[frame] = np.mean(feet, axis=0)

    raw_error = np.linalg.norm(com[:, :2] - support_center, axis=1)
    error = moving_average(raw_error, 9)
    single = np.logical_xor(left_contact, right_contact)
    return {
        "support_error_mean": float(error.mean()),
        "support_error_p95": float(np.percentile(error, 95)),
        "support_error_max": float(error.max()),
        "single_support_fraction": float(single.mean()),
        "double_support_fraction": float((left_contact & right_contact).mean()),
    }


def detect_contact_from_positions(pos: np.ndarray, fps: float) -> np.ndarray:
    speed = np.zeros(len(pos), dtype=np.float64)
    speed[1:] = np.linalg.norm(np.diff(pos, axis=0), axis=1) * fps
    low_height = pos[:, 2] <= np.percentile(pos[:, 2], 40) + 0.035
    slow = speed <= np.percentile(speed, 55) + 0.12
    return close_small_gaps(low_height & slow, max_gap=3)


def close_small_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    out = mask.copy()
    start = None
    for index, value in enumerate(~mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if start > 0 and index < len(mask) and index - start <= max_gap:
                out[start:index] = True
            start = None
    if start is not None and start > 0 and len(mask) - start <= max_gap:
        out[start:] = True
    return out


def whole_body_com(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    masses = model.body_mass[1:]
    weighted = data.xipos[1:] * masses[:, None]
    return weighted.sum(axis=0) / max(float(masses.sum()), 1e-9)


def support_com_score(rollout: dict[str, float | None], support: dict[str, float],
                      expressiveness: float, min_stability: float) -> float:
    stability = float(rollout["stability_rate"])
    tracking = float(rollout["mean_joint_error"])
    pelvis_z = float(rollout["pelvis_z_min"])
    support_mean = float(support["support_error_mean"])
    support_p95 = float(support["support_error_p95"])

    fall_penalty = 260.0 * max(0.0, min_stability - stability) ** 2 + 70.0 * (1.0 - stability)
    height_penalty = 8.0 * max(0.0, 0.62 - pelvis_z)
    tracking_penalty = 2.2 * tracking
    support_penalty = 5.0 * support_mean + 8.0 * max(0.0, support_p95 - 0.10)
    expression_reward = 4.8 * expressiveness if stability >= min_stability else 0.7 * expressiveness
    return fall_penalty + height_penalty + tracking_penalty + support_penalty - expression_reward


if __name__ == "__main__":
    main()
