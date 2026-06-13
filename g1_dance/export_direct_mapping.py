"""Export a direct SMPL-to-G1 qpos reference.

This baseline bypasses v2 arm IK, offline filtering, feasibility optimization,
and showcase gain amplification. It is useful for checking the clean mapping
behavior of joints that have obvious SMPL/G1 correspondences.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import mujoco
import numpy as np

from g1_dance.project_paths import G1_XML, OPTIMIZED_MOTIONS_DIR
from g1_dance.retarget_smpl_to_g1 import load_motion, smpl_to_g1_qpos


def save_direct_reference(
    out_path: str | os.PathLike,
    qpos: np.ndarray,
    fps: float,
    source: str,
) -> None:
    """Save a G1 qpos reference using the same field names as other exporters."""
    np.savez_compressed(
        out_path,
        qpos_ref=qpos.astype(np.float32),
        fps=np.array([fps], dtype=np.float32),
        source=np.array([source]),
        mapping=np.array(["direct_dof"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export direct SMPL-to-G1 qpos reference")
    parser.add_argument("motion", help="Input SMPL .pkl motion")
    parser.add_argument("--fps", type=float, default=60.0, help="Source motion fps")
    parser.add_argument("--out", help="Output .npz path")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(G1_XML)
    poses, trans = load_motion(args.motion)
    qpos = smpl_to_g1_qpos(model, poses, trans)

    OPTIMIZED_MOTIONS_DIR.mkdir(exist_ok=True)
    out_path = Path(args.out) if args.out else OPTIMIZED_MOTIONS_DIR / f"{Path(args.motion).stem}_direct.npz"
    save_direct_reference(out_path, qpos, args.fps, args.motion)
    print(f"Wrote {out_path}")
    print(f"qpos_ref={qpos.shape} fps={args.fps:g}")


if __name__ == "__main__":
    main()
