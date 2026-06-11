"""Offline constraints and trajectory cleanup for G1 dance references.

This module performs the first stage of a dynamics-aware pipeline: it does not
solve full MuJoCo dynamics, but it makes the retargeted qpos trajectory much
more robot-friendly before rollout by smoothing joints, limiting velocities,
stabilizing the floating base, and reducing foot slip during contact phases.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R


LEFT_FOOT_BODY = "left_ankle_roll_link"
RIGHT_FOOT_BODY = "right_ankle_roll_link"


@dataclass
class OptimizationReport:
    frames: int
    fps: float
    left_contact_frames: int
    right_contact_frames: int
    root_xy_change_mean: float
    max_joint_step_before: float
    max_joint_step_after: float


class JointProfile(NamedTuple):
    window: int
    max_step: float
    max_acc: float
    gain: float


def joint_addresses(model) -> dict[str, int]:
    return {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i): int(model.jnt_qposadr[i])
        for i in range(model.njnt)
    }


def actuator_qpos_addresses(model) -> np.ndarray:
    out = np.zeros(model.nu, dtype=np.int64)
    for i in range(model.nu):
        joint_id = model.actuator_trnid[i, 0]
        out[i] = model.jnt_qposadr[joint_id]
    return out


def body_positions(model, qpos: np.ndarray, body_names: list[str]) -> dict[str, np.ndarray]:
    data = mujoco.MjData(model)
    body_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in body_names
    }
    for name, body_id in body_ids.items():
        if body_id < 0:
            raise ValueError(f"Missing body: {name}")

    positions = {name: np.zeros((qpos.shape[0], 3), dtype=np.float64) for name in body_names}
    for frame, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        for name, body_id in body_ids.items():
            positions[name][frame] = data.xpos[body_id]
    return positions


def detect_foot_contacts(model, qpos: np.ndarray, fps: float) -> dict[str, np.ndarray]:
    positions = body_positions(model, qpos, [LEFT_FOOT_BODY, RIGHT_FOOT_BODY])
    contacts = {}
    for key, pos in [("left", positions[LEFT_FOOT_BODY]), ("right", positions[RIGHT_FOOT_BODY])]:
        speed = np.zeros(len(pos), dtype=np.float64)
        speed[1:] = np.linalg.norm(np.diff(pos, axis=0), axis=1) * fps
        low_height = pos[:, 2] <= np.percentile(pos[:, 2], 35) + 0.035
        slow = speed <= np.percentile(speed, 45) + 0.15
        contacts[key] = _close_small_gaps(low_height & slow, max_gap=3)
    return contacts


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < 3:
        return values.copy()
    window = min(window, len(values) if len(values) % 2 else len(values) - 1)
    if window <= 1:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")


def rate_limit(values: np.ndarray, max_step: float) -> np.ndarray:
    out = values.copy()
    for i in range(1, len(out)):
        delta = np.clip(out[i] - out[i - 1], -max_step, max_step)
        out[i] = out[i - 1] + delta
    for i in range(len(out) - 2, -1, -1):
        delta = np.clip(out[i] - out[i + 1], -max_step, max_step)
        out[i] = out[i + 1] + delta
    return out


def acceleration_limit(values: np.ndarray, max_acc_step: float, passes: int = 2) -> np.ndarray:
    out = values.copy()
    for _ in range(passes):
        for i in range(1, len(out) - 1):
            predicted = 0.5 * (out[i - 1] + out[i + 1])
            out[i] = predicted + np.clip(out[i] - predicted, -max_acc_step, max_acc_step)
    return out


def optimize_qpos_offline(
    model,
    qpos_init: np.ndarray,
    fps: float,
    profile: str = "safe",
) -> tuple[np.ndarray, dict[str, np.ndarray], OptimizationReport]:
    qpos = qpos_init.astype(np.float64).copy()
    before = _max_actuated_step(model, qpos)
    root_xy_before = qpos[:, :2].copy()
    addresses = joint_addresses(model)

    _stabilize_root(qpos)
    _smooth_actuated_joints(model, qpos, addresses, profile=profile)
    if profile == "balance":
        _project_balance_root(qpos)

    contacts = detect_foot_contacts(model, qpos, fps)
    for _ in range(3):
        _lock_support_feet(model, qpos, contacts)
        _stabilize_root(qpos, light=True)

    _clip_joint_ranges(model, qpos)
    after = _max_actuated_step(model, qpos)
    report = OptimizationReport(
        frames=qpos.shape[0],
        fps=fps,
        left_contact_frames=int(contacts["left"].sum()),
        right_contact_frames=int(contacts["right"].sum()),
        root_xy_change_mean=float(np.linalg.norm(qpos[:, :2] - root_xy_before, axis=1).mean()),
        max_joint_step_before=before,
        max_joint_step_after=after,
    )
    return qpos, contacts, report


def _stabilize_root(qpos: np.ndarray, light: bool = False) -> None:
    quat_xyzw = qpos[:, [4, 5, 6, 3]]
    euler = R.from_quat(quat_xyzw).as_euler("xyz", degrees=False)
    euler[:, 0] *= 0.10 if light else 0.03
    euler[:, 1] *= 0.10 if light else 0.03
    euler[:, 2] = moving_average(np.unwrap(euler[:, 2]), 11 if light else 17)
    quat = R.from_euler("xyz", euler).as_quat()
    qpos[:, 3:7] = quat[:, [3, 0, 1, 2]]
    qpos[:, 2] = moving_average(qpos[:, 2], 11 if light else 17)


def _project_balance_root(qpos: np.ndarray) -> None:
    """Keep the floating base near a standing root for free-root rollouts."""
    qpos[:, 0] = qpos[0, 0]
    qpos[:, 1] = qpos[0, 1]
    z_center = float(np.mean(qpos[:, 2]))
    qpos[:, 2] = z_center + 0.15 * (qpos[:, 2] - z_center)


def _smooth_actuated_joints(
    model,
    qpos: np.ndarray,
    addresses: dict[str, int],
    profile: str = "safe",
) -> None:
    for joint_name, address in addresses.items():
        if address < 7:
            continue
        window, max_step, max_acc, gain = _joint_profile(joint_name, profile=profile)
        values = qpos[:, address] * gain
        values = moving_average(values, window)
        values = acceleration_limit(values, max_acc)
        qpos[:, address] = rate_limit(values, max_step)


def _joint_profile(joint_name: str, profile: str = "safe") -> JointProfile:
    if profile == "safe":
        if "wrist" in joint_name:
            return JointProfile(15, 0.055, 0.025, 0.08)
        if "shoulder" in joint_name:
            return JointProfile(13, 0.085, 0.035, 0.62)
        if "elbow" in joint_name:
            return JointProfile(13, 0.075, 0.030, 0.55)
        if "waist" in joint_name:
            return JointProfile(15, 0.060, 0.025, 0.45)
        if "ankle" in joint_name:
            return JointProfile(9, 0.080, 0.035, 0.70)
        if "hip" in joint_name:
            return JointProfile(9, 0.090, 0.040, 0.75)
        if "knee" in joint_name:
            return JointProfile(9, 0.085, 0.040, 0.70)
        return JointProfile(11, 0.080, 0.035, 0.70)

    if profile == "expressive":
        if "wrist" in joint_name:
            return JointProfile(13, 0.060, 0.025, 0.16)
        if "shoulder" in joint_name:
            return JointProfile(9, 0.120, 0.050, 0.82)
        if "elbow" in joint_name:
            return JointProfile(9, 0.095, 0.040, 0.70)
        if "waist" in joint_name:
            return JointProfile(11, 0.075, 0.032, 0.62)
        if "ankle" in joint_name:
            return JointProfile(9, 0.085, 0.035, 0.72)
        if "hip" in joint_name:
            return JointProfile(9, 0.095, 0.040, 0.78)
        if "knee" in joint_name:
            return JointProfile(9, 0.090, 0.040, 0.74)
        return JointProfile(9, 0.090, 0.035, 0.75)

    if profile == "balance":
        if "wrist" in joint_name:
            return JointProfile(15, 0.040, 0.018, 0.06)
        if "shoulder" in joint_name:
            return JointProfile(13, 0.050, 0.022, 0.35)
        if "elbow" in joint_name:
            return JointProfile(13, 0.050, 0.022, 0.35)
        if "waist" in joint_name:
            return JointProfile(15, 0.035, 0.016, 0.08)
        if "ankle" in joint_name:
            return JointProfile(15, 0.035, 0.016, 0.05)
        if "hip" in joint_name:
            return JointProfile(15, 0.035, 0.016, 0.05)
        if "knee" in joint_name:
            return JointProfile(15, 0.035, 0.016, 0.05)
        return JointProfile(15, 0.040, 0.018, 0.20)

    if profile == "showcase":
        if "wrist" in joint_name:
            return JointProfile(9, 0.075, 0.030, 0.22)
        if "shoulder" in joint_name:
            return JointProfile(7, 0.135, 0.055, 0.95)
        if "elbow" in joint_name:
            return JointProfile(7, 0.110, 0.045, 0.82)
        if "waist" in joint_name:
            return JointProfile(9, 0.080, 0.034, 0.50)
        if "ankle" in joint_name:
            return JointProfile(11, 0.070, 0.030, 0.45)
        if "hip" in joint_name:
            return JointProfile(11, 0.075, 0.032, 0.48)
        if "knee" in joint_name:
            return JointProfile(11, 0.075, 0.032, 0.48)
        return JointProfile(9, 0.085, 0.035, 0.65)

    raise ValueError(f"Unknown constraint profile: {profile}")


def _lock_support_feet(model, qpos: np.ndarray, contacts: dict[str, np.ndarray]) -> None:
    body_map = {"left": LEFT_FOOT_BODY, "right": RIGHT_FOOT_BODY}
    positions = body_positions(model, qpos, list(body_map.values()))
    for side, body_name in body_map.items():
        mask = contacts[side]
        for start, end in _segments(mask):
            if end - start < 3:
                continue
            target = positions[body_name][start, :2].copy()
            for frame in range(start + 1, end):
                current = positions[body_name][frame, :2]
                qpos[frame, :2] += 0.55 * (target - current)


def _segments(mask: np.ndarray):
    start = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            yield start, i
            start = None
    if start is not None:
        yield start, len(mask)


def _close_small_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    out = mask.copy()
    for start, end in _segments(~mask):
        if start > 0 and end < len(mask) and end - start <= max_gap:
            out[start:end] = True
    return out


def _clip_joint_ranges(model, qpos: np.ndarray) -> None:
    for joint_id in range(1, model.njnt):
        low, high = model.jnt_range[joint_id]
        if low < high:
            address = int(model.jnt_qposadr[joint_id])
            margin = 0.03 * (high - low)
            qpos[:, address] = np.clip(qpos[:, address], low + margin, high - margin)


def _max_actuated_step(model, qpos: np.ndarray) -> float:
    addresses = actuator_qpos_addresses(model)
    if len(qpos) < 2:
        return 0.0
    return float(np.abs(np.diff(qpos[:, addresses], axis=0)).max())
