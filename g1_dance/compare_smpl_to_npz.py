"""Render original SMPL motion next to a saved G1 qpos reference."""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np

from g1_dance.project_paths import G1_XML, VIDEOS_DIR
from g1_dance.retarget_smpl_to_g1 import load_motion
from g1_dance.sidebyside import render_smpl_frame
from g1_dance.smpl_fk import fk


W = H = 480


def output_path_for_comparison(motion_path: str, ref_path: str) -> str:
    """Return the default output video path for a SMPL-vs-G1 comparison."""
    motion_stem = Path(motion_path).stem
    ref_stem = Path(ref_path).stem
    return str(VIDEOS_DIR / f"{motion_stem}_vs_{ref_stem}_compare.mp4")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render SMPL source vs G1 npz reference")
    parser.add_argument("motion", help="Original SMPL .pkl motion")
    parser.add_argument("ref", help="G1 reference .npz containing qpos_ref")
    parser.add_argument("--fps", type=float, default=60.0, help="Source motion fps")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Limit output frames at 30 fps; 0 renders all",
    )
    parser.add_argument("--out", help="Output .mp4 path")
    args = parser.parse_args()

    poses, trans = load_motion(args.motion)
    joints3d = fk(poses, trans)
    ref = np.load(args.ref, allow_pickle=True)
    qpos = np.asarray(ref["qpos_ref"], dtype=np.float64)

    model = mujoco.MjModel.from_xml_path(G1_XML)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=H, width=W)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.distance = 4.0
    cam.azimuth = 135
    cam.elevation = -15

    step = max(1, int(round(args.fps / 30.0)))
    n_out = min(poses.shape[0], qpos.shape[0]) // step
    if args.max_frames > 0:
        n_out = min(n_out, args.max_frames)

    frames = []
    for k in range(n_out):
        t = k * step
        smpl_img = render_smpl_frame(joints3d[t], np.array([trans[t, 0], trans[t, 2]]))

        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        cam.lookat[:] = data.qpos[0:3]
        renderer.update_scene(data, camera=cam)
        g1_img = renderer.render()
        frames.append(np.concatenate([smpl_img, g1_img], axis=1))
        if k % 60 == 0:
            print(f"  {k}/{n_out}")

    VIDEOS_DIR.mkdir(exist_ok=True)
    out_path = args.out or output_path_for_comparison(args.motion, args.ref)
    imageio.mimwrite(out_path, frames, fps=30, quality=8)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
