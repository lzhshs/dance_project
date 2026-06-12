"""Contact-aware root balance correction for G1 dance references.

This post-processing stage tries to bridge the gap between pinned-root showcase
motions and conservative free-root balance motions. It keeps upper-body motion
from the input reference, damps lower-body/root motion, estimates a support
center from contacting feet, and applies a small PD-style correction to root xy.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from motion_constraints import (
    LEFT_FOOT_BODY,
    RIGHT_FOOT_BODY,
    body_positions,
    detect_foot_contacts,
    joint_addresses,
    moving_average,
    rate_limit,
)
from project_paths import G1_XML, OPTIMIZED_MOTIONS_DIR


def compute_support_centers(
    left_xy: np.ndarray,
    right_xy: np.ndarray,
    left_contact: np.ndarray,
    right_contact: np.ndarray,
    fallback: np.ndarray | None = None,
) -> np.ndarray:
    """Return a support-center xy trajectory from foot contact masks."""
    centers = np.zeros_like(left_xy, dtype=np.float64)
    previous = (
        np.asarray(fallback, dtype=np.float64).copy()
        if fallback is not None
        else 0.5 * (left_xy[0] + right_xy[0])
    )
    for i in range(len(centers)):
        if left_contact[i] and right_contact[i]:
            previous = 0.5 * (left_xy[i] + right_xy[i])
        elif left_contact[i]:
            previous = left_xy[i].copy()
        elif right_contact[i]:
            previous = right_xy[i].copy()
        centers[i] = previous
    return centers


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


def compute_segment_median_support_centers(
    left_xy: np.ndarray,
    right_xy: np.ndarray,
    left_contact: np.ndarray,
    right_contact: np.ndarray,
    fallback: np.ndarray | None = None,
) -> np.ndarray:
    """Use one median support point per contact segment to reduce foot-slip drift."""
    centers = compute_support_centers(left_xy, right_xy, left_contact, right_contact, fallback=fallback)

    both = left_contact & right_contact
    left_only = left_contact & ~right_contact
    right_only = right_contact & ~left_contact
    for mask, values in [
        (both, 0.5 * (left_xy + right_xy)),
        (left_only, left_xy),
        (right_only, right_xy),
    ]:
        for start, end in _segments(mask):
            if end - start < 2:
                continue
            centers[start:end] = np.median(values[start:end], axis=0)

    previous = centers[0].copy()
    for i in range(len(centers)):
        if left_contact[i] or right_contact[i]:
            previous = centers[i].copy()
        else:
            centers[i] = previous
    return centers


def pd_correct_root_xy(
    qpos: np.ndarray,
    support_xy: np.ndarray,
    kp: float,
    kd: float,
    max_step: float,
) -> np.ndarray:
    """Apply a clipped PD-style root xy correction toward the support center."""
    out = qpos.astype(np.float64).copy()
    for i in range(1, len(out)):
        velocity = out[i, :2] - out[i - 1, :2]
        error = support_xy[i] - out[i, :2]
        correction = kp * error - kd * velocity
        norm = float(np.linalg.norm(correction))
        if norm > max_step:
            correction *= max_step / max(norm, 1e-9)
        out[i, :2] += correction
    out[:, 0] = rate_limit(moving_average(out[:, 0], 9), max_step)
    out[:, 1] = rate_limit(moving_average(out[:, 1], 9), max_step)
    return out


def limit_root_motion(qpos: np.ndarray, gain: float) -> np.ndarray:
    """Scale root xy displacement around the first frame."""
    out = qpos.astype(np.float64).copy()
    out[:, :2] = out[0, :2] + gain * (out[:, :2] - out[0, :2])
    return out


def damp_lower_body(
    model,
    qpos: np.ndarray,
    leg_gain: float,
    waist_gain: float,
    arm_gain: float = 1.0,
) -> None:
    """Damp joint groups before contact-aware root correction."""
    addresses = joint_addresses(model)
    for joint_name, address in addresses.items():
        if address < 7:
            continue
        if any(token in joint_name for token in ("hip", "knee", "ankle")):
            qpos[:, address] = rate_limit(moving_average(qpos[:, address] * leg_gain, 9), 0.060)
        elif "waist" in joint_name:
            qpos[:, address] = rate_limit(moving_average(qpos[:, address] * waist_gain, 11), 0.045)
        elif any(token in joint_name for token in ("shoulder", "elbow", "wrist")) and arm_gain < 0.999:
            qpos[:, address] = rate_limit(moving_average(qpos[:, address] * arm_gain, 9), 0.080)


def stabilize_root_pose(qpos: np.ndarray, z_gain: float, roll_pitch_gain: float) -> None:
    """Damp root height variation and roll/pitch while preserving yaw."""
    quat_xyzw = qpos[:, [4, 5, 6, 3]]
    euler = R.from_quat(quat_xyzw).as_euler("xyz", degrees=False)
    euler[:, 0] *= roll_pitch_gain
    euler[:, 1] *= roll_pitch_gain
    euler[:, 2] = moving_average(np.unwrap(euler[:, 2]), 11)
    quat = R.from_euler("xyz", euler).as_quat()
    qpos[:, 3:7] = quat[:, [3, 0, 1, 2]]

    z_center = float(np.mean(qpos[:, 2]))
    qpos[:, 2] = z_center + z_gain * (qpos[:, 2] - z_center)
    qpos[:, 2] = moving_average(qpos[:, 2], 9)


def contact_balance_reference(
    model,
    qpos_ref: np.ndarray,
    fps: float,
    root_kp: float = 0.18,
    root_kd: float = 0.08,
    max_root_step: float = 0.025,
    root_motion_gain: float = 0.35,
    leg_gain: float = 0.60,
    waist_gain: float = 0.55,
    arm_gain: float = 1.0,
    root_z_gain: float = 0.30,
    root_roll_pitch_gain: float = 0.08,
    stable_support: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    """Build a contact-aware balance-corrected qpos reference."""
    qpos = qpos_ref.astype(np.float64).copy()
    contacts = detect_foot_contacts(model, qpos, fps)
    foot_pos = body_positions(model, qpos, [LEFT_FOOT_BODY, RIGHT_FOOT_BODY])
    support_fn = compute_segment_median_support_centers if stable_support else compute_support_centers
    support = support_fn(
        foot_pos[LEFT_FOOT_BODY][:, :2],
        foot_pos[RIGHT_FOOT_BODY][:, :2],
        contacts["left"],
        contacts["right"],
        fallback=qpos[0, :2],
    )
    support[:, 0] = moving_average(support[:, 0], 9)
    support[:, 1] = moving_average(support[:, 1], 9)

    damp_lower_body(model, qpos, leg_gain=leg_gain, waist_gain=waist_gain, arm_gain=arm_gain)
    stabilize_root_pose(qpos, z_gain=root_z_gain, roll_pitch_gain=root_roll_pitch_gain)
    qpos = pd_correct_root_xy(qpos, support, kp=root_kp, kd=root_kd, max_step=max_root_step)
    qpos = limit_root_motion(qpos, gain=root_motion_gain)
    stats = {
        "root_xy_range_before": float(np.ptp(qpos_ref[:, :2], axis=0).max()),
        "root_xy_range_after": float(np.ptp(qpos[:, :2], axis=0).max()),
        "mean_joint_std_after": float(np.std(qpos[:, 7:], axis=0).mean()),
        "left_contact_frames": float(contacts["left"].sum()),
        "right_contact_frames": float(contacts["right"].sum()),
        "stable_support": float(stable_support),
    }
    return qpos, contacts, stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply contact-aware PD balance correction")
    parser.add_argument("ref", help="Input .npz containing qpos_ref")
    parser.add_argument("--out", help="Output .npz path")
    parser.add_argument("--root-kp", type=float, default=0.18)
    parser.add_argument("--root-kd", type=float, default=0.08)
    parser.add_argument("--max-root-step", type=float, default=0.025)
    parser.add_argument("--root-motion-gain", type=float, default=0.35)
    parser.add_argument("--leg-gain", type=float, default=0.60)
    parser.add_argument("--waist-gain", type=float, default=0.55)
    parser.add_argument("--arm-gain", type=float, default=1.0)
    parser.add_argument("--root-z-gain", type=float, default=0.30)
    parser.add_argument("--root-roll-pitch-gain", type=float, default=0.08)
    parser.add_argument("--stable-support", action="store_true",
                        help="Use median support points over contact segments")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source = np.load(args.ref, allow_pickle=True)
    qpos_ref = np.asarray(source["qpos_ref"], dtype=np.float64)
    fps = float(source["fps"][0]) if "fps" in source else 30.0

    model = mujoco.MjModel.from_xml_path(G1_XML)
    qpos, contacts, stats = contact_balance_reference(
        model,
        qpos_ref,
        fps,
        root_kp=args.root_kp,
        root_kd=args.root_kd,
        max_root_step=args.max_root_step,
        root_motion_gain=args.root_motion_gain,
        leg_gain=args.leg_gain,
        waist_gain=args.waist_gain,
        arm_gain=args.arm_gain,
        root_z_gain=args.root_z_gain,
        root_roll_pitch_gain=args.root_roll_pitch_gain,
        stable_support=args.stable_support,
    )

    OPTIMIZED_MOTIONS_DIR.mkdir(exist_ok=True)
    stem = Path(args.ref).stem
    out_path = args.out or os.path.join(OPTIMIZED_MOTIONS_DIR, f"{stem}_contact_balance.npz")
    np.savez_compressed(
        out_path,
        qpos_ref=qpos.astype(np.float32),
        fps=np.array([fps], dtype=np.float32),
        source=np.array([args.ref]),
        left_contact=contacts["left"],
        right_contact=contacts["right"],
        profile=np.array(["contact_balance"]),
        contact_balance_stats=json.dumps(stats),
    )
    print(f"Wrote {out_path}")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
