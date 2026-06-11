"""Robot-feasibility optimizer for G1 dance references.

This stage sits after the offline retargeting/filtering stage.  It searches for
the largest body-group motion gains that remain feasible in an unpinned MuJoCo
rollout, so the result is less conservative than ``make_balance_safe.py`` while
still explicitly optimizing for not falling.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from motion_constraints import actuator_qpos_addresses, joint_addresses, moving_average, rate_limit
from retarget_smpl_to_g1 import G1_XML
from rollout_optimize import FALL_PELVIS_Z, interp_qpos


OUT_DIR = "/Users/lucy_lzh/dance_project/optimized_motions"


@dataclass(frozen=True)
class FeasibilityParams:
    leg_gain: float
    arm_gain: float
    waist_gain: float
    root_xy_gain: float
    root_yaw_gain: float
    vertical_gain: float
    smooth_scale: float


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize a G1 reference for MuJoCo feasibility")
    parser.add_argument("ref", help="Input .npz from optimize_g1_motion.py")
    parser.add_argument("--out", help="Output .npz path")
    parser.add_argument("--max-seconds", type=float, default=10.0)
    parser.add_argument("--samples", type=int, default=90, help="Number of parameter samples to test")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-stability", type=float, default=0.98)
    args = parser.parse_args()

    source = np.load(args.ref, allow_pickle=True)
    qpos_source = np.asarray(source["qpos_ref"], dtype=np.float64)
    fps = float(source["fps"][0]) if "fps" in source else 30.0
    model = mujoco.MjModel.from_xml_path(G1_XML)

    params = build_parameter_set(args.samples, args.seed)
    print(f"Testing {len(params)} robot-feasibility candidates ...")

    results = []
    for index, param in enumerate(params, start=1):
        candidate = make_feasible_reference(model, qpos_source, param)
        stats = simulate_feasibility(model, candidate, fps, args.max_seconds)
        expressiveness = motion_expressiveness(model, candidate, qpos_source)
        score = feasibility_score(stats, expressiveness, args.min_stability)
        results.append((score, expressiveness, param, candidate, stats))
        print(
            f"{index:03d}/{len(params):03d} score={score:8.3f} "
            f"stable={stats['stability_rate'] * 100:5.1f}% "
            f"expr={expressiveness:5.3f} "
            f"leg={param.leg_gain:.2f} arm={param.arm_gain:.2f} "
            f"waist={param.waist_gain:.2f} root={param.root_xy_gain:.2f}"
        )

    score, expressiveness, best_param, best, stats = min(results, key=lambda item: item[0])
    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.ref))[0].replace("_optimized", "")
    out_path = args.out or os.path.join(OUT_DIR, f"{stem}_feasible.npz")
    report = {
        "params": asdict(best_param),
        "score": score,
        "expressiveness": expressiveness,
        "stats": stats,
        "min_stability": args.min_stability,
    }
    np.savez_compressed(
        out_path,
        qpos_ref=best.astype(np.float32),
        qpos_source=qpos_source.astype(np.float32),
        fps=np.array([fps], dtype=np.float32),
        source=np.array([args.ref]),
        feasibility_report=json.dumps(report, ensure_ascii=False),
    )
    print("\nSelected robot-feasible reference")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote {out_path}")


def build_parameter_set(samples: int, seed: int) -> list[FeasibilityParams]:
    rng = np.random.default_rng(seed)
    presets = [
        FeasibilityParams(0.00, 0.35, 0.03, 0.00, 0.15, 0.05, 1.25),
        FeasibilityParams(0.06, 0.50, 0.06, 0.02, 0.20, 0.06, 1.10),
        FeasibilityParams(0.12, 0.65, 0.10, 0.04, 0.25, 0.08, 1.00),
        FeasibilityParams(0.18, 0.80, 0.14, 0.06, 0.30, 0.10, 0.95),
        FeasibilityParams(0.25, 0.95, 0.18, 0.08, 0.35, 0.12, 0.90),
        FeasibilityParams(0.35, 1.10, 0.24, 0.10, 0.45, 0.14, 0.85),
    ]
    params = list(presets)
    while len(params) < samples:
        leg_gain = float(rng.beta(1.3, 4.5) * 0.65)
        arm_gain = float(0.35 + rng.beta(2.0, 1.7) * 0.95)
        waist_gain = float(rng.beta(1.4, 4.0) * 0.38)
        root_xy_gain = float(rng.beta(1.2, 5.0) * 0.22)
        root_yaw_gain = float(0.10 + rng.beta(1.4, 3.0) * 0.55)
        vertical_gain = float(rng.beta(1.0, 5.0) * 0.18)
        smooth_scale = float(0.75 + rng.beta(1.5, 2.0) * 0.75)
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


def make_feasible_reference(model: mujoco.MjModel, qpos_source: np.ndarray, param: FeasibilityParams) -> np.ndarray:
    qpos = qpos_source.copy()
    stand = model.key_qpos[0].copy() if model.nkey else qpos_source[0].copy()

    qpos[:, 0] = qpos[0, 0] + param.root_xy_gain * (qpos_source[:, 0] - qpos_source[0, 0])
    qpos[:, 1] = qpos[0, 1] + param.root_xy_gain * (qpos_source[:, 1] - qpos_source[0, 1])
    qpos[:, 2] = stand[2] + param.vertical_gain * moving_average(qpos_source[:, 2] - np.mean(qpos_source[:, 2]), 15)
    qpos[:, 3:7] = feasible_root_quat(qpos_source[:, 3:7], param.root_yaw_gain)

    for joint_name, address in joint_addresses(model).items():
        if address < 7:
            continue
        gain = gain_for_joint(joint_name, param)
        values = stand[address] + gain * (qpos_source[:, address] - stand[address])
        window, max_step = filter_profile(joint_name, param.smooth_scale)
        values = moving_average(values, window)
        qpos[:, address] = rate_limit(values, max_step)

    clip_joint_ranges(model, qpos)
    return qpos


def feasible_root_quat(root_quat_wxyz: np.ndarray, yaw_gain: float) -> np.ndarray:
    euler = R.from_quat(root_quat_wxyz[:, [1, 2, 3, 0]]).as_euler("xyz", degrees=False)
    euler[:, 0] *= 0.02
    euler[:, 1] *= 0.02
    euler[:, 2] = moving_average(np.unwrap(euler[:, 2]) * yaw_gain, 17)
    quat_xyzw = R.from_euler("xyz", euler).as_quat()
    return quat_xyzw[:, [3, 0, 1, 2]]


def gain_for_joint(joint_name: str, param: FeasibilityParams) -> float:
    if any(token in joint_name for token in ["hip", "knee", "ankle"]):
        return param.leg_gain
    if "waist" in joint_name:
        return param.waist_gain
    if any(token in joint_name for token in ["shoulder", "elbow"]):
        return param.arm_gain
    if "wrist" in joint_name:
        return 0.18 * param.arm_gain
    return 0.5


def filter_profile(joint_name: str, smooth_scale: float) -> tuple[int, float]:
    if "wrist" in joint_name:
        base_window, base_step = 17, 0.040
    elif any(token in joint_name for token in ["shoulder", "elbow"]):
        base_window, base_step = 11, 0.070
    elif "waist" in joint_name:
        base_window, base_step = 15, 0.045
    elif any(token in joint_name for token in ["hip", "knee", "ankle"]):
        base_window, base_step = 13, 0.050
    else:
        base_window, base_step = 13, 0.050
    window = int(round(base_window * smooth_scale))
    if window % 2 == 0:
        window += 1
    return max(3, window), base_step / max(0.65, smooth_scale)


def clip_joint_ranges(model: mujoco.MjModel, qpos: np.ndarray) -> None:
    for joint_id in range(1, model.njnt):
        low, high = model.jnt_range[joint_id]
        if low < high:
            address = int(model.jnt_qposadr[joint_id])
            margin = 0.025 * (high - low)
            qpos[:, address] = np.clip(qpos[:, address], low + margin, high - margin)


def simulate_feasibility(model: mujoco.MjModel, qpos_ref: np.ndarray, fps: float, max_seconds: float) -> dict[str, float | None]:
    data = mujoco.MjData(model)
    ctrl_to_qadr = actuator_qpos_addresses(model)
    duration = min(qpos_ref.shape[0] / fps, max_seconds)
    dt = model.opt.timestep
    n_steps = int(duration / dt)

    data.qpos[:] = qpos_ref[0]
    data.qpos[2] += 0.02
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


def motion_expressiveness(model: mujoco.MjModel, qpos_ref: np.ndarray, qpos_source: np.ndarray) -> float:
    addresses = joint_addresses(model)
    weights = np.zeros(model.nq, dtype=np.float64)
    for joint_name, address in addresses.items():
        if address < 7:
            continue
        if any(token in joint_name for token in ["shoulder", "elbow"]):
            weights[address] = 1.00
        elif "waist" in joint_name:
            weights[address] = 0.85
        elif any(token in joint_name for token in ["hip", "knee", "ankle"]):
            weights[address] = 0.55
        elif "wrist" in joint_name:
            weights[address] = 0.20
    active = weights > 0
    source_scale = float(np.mean(np.std(qpos_source[:, active], axis=0) * weights[active])) + 1e-6
    candidate_scale = float(np.mean(np.std(qpos_ref[:, active], axis=0) * weights[active]))
    velocity_scale = float(np.mean(np.abs(np.diff(qpos_ref[:, active], axis=0))) * 30.0)
    return candidate_scale / source_scale + 0.15 * velocity_scale


def feasibility_score(stats: dict[str, float | None], expressiveness: float, min_stability: float) -> float:
    stability_rate = float(stats["stability_rate"])
    tracking_error = float(stats["mean_joint_error"])
    pelvis_z_min = float(stats["pelvis_z_min"])
    fall_penalty = 250.0 * max(0.0, min_stability - stability_rate) ** 2
    full_fall_penalty = 60.0 * (1.0 - stability_rate)
    height_penalty = 8.0 * max(0.0, 0.62 - pelvis_z_min)
    tracking_penalty = 2.5 * tracking_error
    expression_reward = 4.0 * expressiveness if stability_rate >= min_stability else 0.8 * expressiveness
    return fall_penalty + full_fall_penalty + height_penalty + tracking_penalty - expression_reward


if __name__ == "__main__":
    main()
