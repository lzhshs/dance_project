"""Optimize a retargeted G1 dance trajectory before MuJoCo rollout.

Stage 1 of the long-term pipeline: build the v2 retargeting initial guess, apply
offline robot constraints, then save an optimized reference trajectory as NPZ.
"""
from __future__ import annotations

import argparse
import json
import os

import mujoco
import numpy as np

from g1_dance.motion_constraints import optimize_qpos_offline
from g1_dance.retarget_smpl_to_g1 import G1_XML, load_motion
from g1_dance.retarget_v2 import build_qpos_trajectory


from g1_dance.project_paths import OPTIMIZED_MOTIONS_DIR

OUT_DIR = str(OPTIMIZED_MOTIONS_DIR)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline-optimize a G1 qpos trajectory")
    parser.add_argument("motion", help="Input SMPL motion pkl")
    parser.add_argument("--fps", type=float, default=0, help="Input motion fps")
    parser.add_argument("--out", help="Output .npz path")
    parser.add_argument(
        "--profile",
        choices=["safe", "expressive", "balance", "showcase"],
        default="safe",
        help=(
            "Constraint profile: safe preserves previous behavior; expressive "
            "keeps more torso/arm motion; balance favors free-root stability; "
            "showcase emphasizes upper-body motion for pinned-root demos"
        ),
    )
    return parser


def infer_fps(path: str, explicit: float) -> float:
    if explicit > 0:
        return explicit
    return 30.0 if "generated_motions" in path else 60.0


def main():
    parser = build_parser()
    args = parser.parse_args()

    fps = infer_fps(args.motion, args.fps)
    stem = os.path.splitext(os.path.basename(args.motion))[0]
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = args.out or os.path.join(OUT_DIR, f"{stem}_optimized.npz")

    print(f"Loading {args.motion}  fps={fps:g}")
    poses, trans = load_motion(args.motion)
    model = mujoco.MjModel.from_xml_path(G1_XML)
    data = mujoco.MjData(model)

    print("Building v2 retargeting initial guess ...")
    qpos_init = build_qpos_trajectory(model, data, poses, trans)

    print(f"Applying offline robot constraints ({args.profile}) ...")
    qpos_ref, contacts, report = optimize_qpos_offline(model, qpos_init, fps, profile=args.profile)

    np.savez_compressed(
        out_path,
        qpos_ref=qpos_ref.astype(np.float32),
        qpos_init=qpos_init.astype(np.float32),
        fps=np.array([fps], dtype=np.float32),
        left_contact=contacts["left"],
        right_contact=contacts["right"],
        source=np.array([args.motion]),
        profile=np.array([args.profile]),
    )

    print(f"Wrote {out_path}")
    print(json.dumps(report.__dict__, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
