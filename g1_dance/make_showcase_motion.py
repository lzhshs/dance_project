"""Create a dance-showcase G1 reference with larger visual motion.

This intentionally favors appearance over strict no-pin physical feasibility.
Use it with `pd_track.py <out.npz> --pin-root` for demo videos where the goal is
to look like a dance rather than prove free-root dynamic executability.
"""
from __future__ import annotations

import argparse
import os

import mujoco
import numpy as np

from g1_dance.motion_constraints import moving_average, rate_limit
from g1_dance.retarget_smpl_to_g1 import G1_XML


from g1_dance.project_paths import OPTIMIZED_MOTIONS_DIR

OUT_DIR = str(OPTIMIZED_MOTIONS_DIR)


def main():
    parser = argparse.ArgumentParser(description="Build a high-amplitude showcase reference")
    parser.add_argument("ref", help="Input .npz, preferably from optimize_g1_motion.py")
    parser.add_argument("--out", help="Output .npz")
    parser.add_argument("--style", choices=["balanced", "expressive", "robotic", "big"], default="expressive")
    args = parser.parse_args()

    source = np.load(args.ref, allow_pickle=True)
    qpos_safe = np.asarray(source["qpos_ref"], dtype=np.float64)
    qpos_init = np.asarray(source["qpos_init"], dtype=np.float64) if "qpos_init" in source else qpos_safe
    fps = float(source["fps"][0]) if "fps" in source else 30.0
    model = mujoco.MjModel.from_xml_path(G1_XML)

    qpos = build_robotic_showcase(model, qpos_init, qpos_safe, fps, big=args.style == "big") \
        if args.style in ("robotic", "big") else \
        build_showcase(model, qpos_init, qpos_safe, style=args.style)
    os.makedirs(OUT_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.ref))[0].replace("_optimized", "")
    out_path = args.out or os.path.join(OUT_DIR, f"{stem}_showcase_{args.style}.npz")
    np.savez_compressed(
        out_path,
        qpos_ref=qpos.astype(np.float32),
        fps=np.array([fps], dtype=np.float32),
        source=np.array([args.ref]),
        style=np.array([args.style]),
    )
    print(f"Wrote {out_path}")
    print("std lower/waist/arms/rootxy:",
          float(qpos[:, 7:19].std()), float(qpos[:, 19:22].std()),
          float(qpos[:, 22:].std()), float(qpos[:, :2].std()))


def build_showcase(model, qpos_init: np.ndarray, qpos_safe: np.ndarray, style: str) -> np.ndarray:
    qpos = qpos_safe.copy()
    if style == "expressive":
        lower_gain, waist_gain, arm_gain, wrist_gain = 0.95, 1.25, 1.45, 0.55
        lower_smooth, arm_smooth = 5, 5
        lower_step, arm_step = 0.16, 0.20
    else:
        lower_gain, waist_gain, arm_gain, wrist_gain = 0.65, 0.90, 1.15, 0.35
        lower_smooth, arm_smooth = 7, 7
        lower_step, arm_step = 0.11, 0.15

    qpos[:, :2] = qpos_init[:, :2]
    qpos[:, 2] = moving_average(0.65 * qpos_init[:, 2] + 0.35 * qpos_safe[:, 2], 7)
    qpos[:, 3:7] = qpos_init[:, 3:7]

    for address in range(7, 19):
        values = lower_gain * qpos_init[:, address] + (1.0 - lower_gain) * qpos_safe[:, address]
        qpos[:, address] = rate_limit(moving_average(values, lower_smooth), lower_step)

    for address in range(19, 22):
        values = waist_gain * qpos_init[:, address]
        qpos[:, address] = rate_limit(moving_average(values, 5), 0.14)

    for address in range(22, qpos.shape[1]):
        gain = wrist_gain if address in (26, 27, 28, 33, 34, 35) else arm_gain
        values = gain * qpos_init[:, address]
        qpos[:, address] = rate_limit(moving_average(values, arm_smooth), arm_step)

    for joint_id in range(1, model.njnt):
        low, high = model.jnt_range[joint_id]
        if low < high:
            address = int(model.jnt_qposadr[joint_id])
            qpos[:, address] = np.clip(qpos[:, address], low, high)
    return qpos


def build_robotic_showcase(model, qpos_init: np.ndarray, qpos_safe: np.ndarray, fps: float, big: bool = False) -> np.ndarray:
    """Make a visibly dance-like robot routine with rhythmic whole-body accents."""
    qpos = np.tile(model.key_qpos[0], (qpos_init.shape[0], 1)).astype(np.float64)
    t = np.arange(qpos.shape[0], dtype=np.float64) / fps
    beat = np.sin(2.0 * np.pi * 1.9 * t)
    beat2 = np.sin(2.0 * np.pi * 0.95 * t + 0.6)
    bounce = 0.5 * (1.0 - np.cos(2.0 * np.pi * 1.9 * t))

    qpos[:, 0] = qpos_init[:, 0]
    qpos[:, 1] = qpos_init[:, 1]
    qpos[:, 2] = 0.79 + 0.035 * bounce
    qpos[:, 3:7] = qpos_init[:, 3:7]

    leg_scale = 1.6 if big else 1.0
    waist_scale = 1.8 if big else 1.0
    arm_scale = 1.8 if big else 1.0

    qpos[:, 7] = -0.08 - 0.10 * leg_scale * bounce
    qpos[:, 10] = 0.12 + 0.18 * leg_scale * bounce
    qpos[:, 11] = -0.04 - 0.06 * leg_scale * bounce
    qpos[:, 13] = -0.08 - 0.10 * leg_scale * bounce
    qpos[:, 16] = 0.12 + 0.18 * leg_scale * bounce
    qpos[:, 17] = -0.04 - 0.06 * leg_scale * bounce

    qpos[:, 19] = 0.32 * waist_scale * beat2
    qpos[:, 20] = 0.13 * waist_scale * beat
    qpos[:, 21] = 0.10 * waist_scale * np.cos(2.0 * np.pi * 1.9 * t)

    qpos[:, 22] = 0.20 + 0.65 * arm_scale * beat
    qpos[:, 23] = 0.70 + 0.38 * arm_scale * beat2
    qpos[:, 24] = 0.35 * arm_scale * np.cos(2.0 * np.pi * 1.9 * t + 0.3)
    qpos[:, 25] = 0.95 + 0.40 * arm_scale * bounce

    qpos[:, 29] = 0.20 - 0.65 * arm_scale * beat
    qpos[:, 30] = -0.70 + 0.38 * arm_scale * beat2
    qpos[:, 31] = -0.35 * arm_scale * np.cos(2.0 * np.pi * 1.9 * t + 0.3)
    qpos[:, 32] = 0.95 + 0.40 * arm_scale * (1.0 - bounce)

    qpos[:, 26] = 0.25 * arm_scale * beat
    qpos[:, 33] = -0.25 * arm_scale * beat

    for address in range(7, qpos.shape[1]):
        qpos[:, address] = rate_limit(moving_average(qpos[:, address], 5), 0.18)

    for joint_id in range(1, model.njnt):
        low, high = model.jnt_range[joint_id]
        if low < high:
            address = int(model.jnt_qposadr[joint_id])
            qpos[:, address] = np.clip(qpos[:, address], low, high)
    return qpos


if __name__ == "__main__":
    main()
