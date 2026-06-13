"""Side-by-side comparison using v2 IK retargeting."""
import os
import sys

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from g1_dance.retarget_smpl_to_g1 import G1_XML, load_motion
from g1_dance.retarget_v2 import build_qpos_trajectory
from g1_dance.smpl_fk import EDGES, fk
from g1_dance.sidebyside import render_smpl_frame

from g1_dance.project_paths import VIDEOS_DIR

OUT_DIR = str(VIDEOS_DIR)
W = H = 480


def main():
    pkl = sys.argv[1]
    stem = os.path.splitext(os.path.basename(pkl))[0]
    out_video = os.path.join(OUT_DIR, stem + "_compare_v2.mp4")

    print(f"Loading {pkl}")
    poses, trans = load_motion(pkl)
    T = poses.shape[0]

    print("SMPL FK ...")
    joints3d = fk(poses, trans)

    print("Retargeting v2 ...")
    model = mujoco.MjModel.from_xml_path(G1_XML)
    data = mujoco.MjData(model)
    qpos_traj = build_qpos_trajectory(model, data, poses, trans)

    renderer = mujoco.Renderer(model, height=H, width=W)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.distance = 4.0
    cam.azimuth = 135
    cam.elevation = -15

    STEP = 2
    n_out = T // STEP
    print(f"Rendering {n_out} frames ...")

    frames = []
    for k in range(n_out):
        t = k * STEP
        data.qpos[:] = qpos_traj[t]
        mujoco.mj_forward(model, data)
        cam.lookat[:] = data.qpos[0:3]
        renderer.update_scene(data, camera=cam)
        g1_img = renderer.render()

        root_xz = np.array([trans[t, 0], trans[t, 2]])
        smpl_img = render_smpl_frame(joints3d[t], root_xz)

        frames.append(np.concatenate([smpl_img, g1_img], axis=1))
        if k % 60 == 0:
            print(f"  {k}/{n_out}")

    print(f"Writing {out_video}")
    imageio.mimwrite(out_video, frames, fps=30, quality=8)
    print("Done.")


if __name__ == "__main__":
    main()
