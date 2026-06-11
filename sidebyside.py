"""Side-by-side comparison: SMPL original skeleton (left) vs G1 retargeted (right).

Usage: python sidebyside.py <motion.pkl>
Output: videos/<stem>_compare.mp4 (+ _compare_audio.mp4 with music)
"""
import os
import sys

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from retarget_smpl_to_g1 import G1_XML, load_motion, smpl_to_g1_qpos
from smpl_fk import EDGES, fk

from project_paths import VIDEOS_DIR

OUT_DIR = str(VIDEOS_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

W = H = 480  # each panel square


def render_smpl_frame(joints3d: np.ndarray, root_xz: np.ndarray) -> np.ndarray:
    """Draw SMPL skeleton as a 3D matplotlib figure and return RGB array."""
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    # joints are Y-up in SMPL
    xs, ys, zs = joints3d[:, 0], joints3d[:, 1], joints3d[:, 2]
    for a, b in EDGES:
        ax.plot([xs[a], xs[b]], [zs[a], zs[b]], [ys[a], ys[b]],
                color="tab:blue", lw=2)
    ax.scatter(xs, zs, ys, c="tab:red", s=10)
    # Follow root (xz in SMPL = floor plane)
    cx, cz = root_xz
    ax.set_xlim(cx - 1.2, cx + 1.2)
    ax.set_ylim(cz - 1.2, cz + 1.2)
    ax.set_zlim(0, 2.2)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=15, azim=135)
    ax.set_axis_off()
    ax.set_title("SMPL (original)")
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return img


def main():
    pkl = sys.argv[1]
    stem = os.path.splitext(os.path.basename(pkl))[0]
    out_video = os.path.join(OUT_DIR, stem + "_compare.mp4")

    print(f"Loading {pkl}")
    poses, trans = load_motion(pkl)
    T = poses.shape[0]

    print("SMPL FK ...")
    joints3d = fk(poses, trans)  # (T, 24, 3) Y-up

    print("Retargeting to G1 ...")
    model = mujoco.MjModel.from_xml_path(G1_XML)
    data = mujoco.MjData(model)
    qpos_traj = smpl_to_g1_qpos(model, poses, trans)

    # MuJoCo renderer
    renderer = mujoco.Renderer(model, height=H, width=W)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.distance = 4.0
    cam.azimuth = 135
    cam.elevation = -15

    STEP = 2  # 60fps -> 30fps
    n_out = T // STEP
    print(f"Rendering {n_out} frames ...")

    frames = []
    for k in range(n_out):
        t = k * STEP
        # G1 panel
        data.qpos[:] = qpos_traj[t]
        mujoco.mj_forward(model, data)
        cam.lookat[:] = data.qpos[0:3]
        renderer.update_scene(data, camera=cam)
        g1_img = renderer.render()

        # SMPL panel
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
