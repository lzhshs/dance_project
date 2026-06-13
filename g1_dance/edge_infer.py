"""Local EDGE inference on Mac using pre-computed Jukebox features.

Workflow:
  1. Extract Jukebox features for your .wav on a GPU box (Colab notebook:
     colab_extract_jukebox.ipynb). You'll get a folder like
       artifacts/cached_features/<song>/
           <song>_slice0.wav
           <song>_slice0.npy
           <song>_slice1.wav
           ...
     Each .npy is a (150, 4800) Jukebox feature array.
  2. Download that folder to `EDGE/cached_features/<song>/` on this Mac.
  3. Run:
       python -m g1_dance.edge_infer <song>
     This loads the features, runs the 50M-param EDGE diffusion on MPS,
     and writes an EDGE-style SMPL .pkl into `artifacts/generated_motions/<song>.pkl`.
  4. Feed into our retarget+PD pipeline:
       python -m g1_dance.retarget_v2 artifacts/generated_motions/<song>.pkl
       python -m g1_dance.pd_track  artifacts/generated_motions/<song>.pkl --pin-root
"""
import glob
import argparse
import os
import pickle
import random
import sys
import tempfile
import types
from functools import cmp_to_key

# Stub jukemirlib so EDGE imports work without it.
sys.modules.setdefault("jukemirlib", types.ModuleType("jukemirlib"))

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from g1_dance.project_paths import CACHED_FEATURES_DIR, EDGE_DIR, GENERATED_MOTIONS_DIR, PROJECT_ROOT

EDGE_DIR = str(EDGE_DIR)
OUT_DIR = str(GENERATED_MOTIONS_DIR)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib_cache"))
sys.path.insert(0, str(PROJECT_ROOT / "vendor"))
sys.path.insert(0, EDGE_DIR)

from EDGE import EDGE  # noqa: E402
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle  # noqa

os.makedirs(OUT_DIR, exist_ok=True)

# EDGE trains/samples in a Z-up coordinate frame: AIST Y-up motions are rotated
# +90 degrees about X during preprocessing. The local retargeting pipeline
# expects AIST-style Y-up SMPL, so convert the sampled root pose/translation
# back before writing the final pkl.
Y_UP_TO_Z_UP = R.from_euler("x", 90, degrees=True)
Z_UP_TO_Y_UP = Y_UP_TO_Z_UP.inv()

# Sort <name>_sliceN.wav/npy by N
def _key(x):
    return int(os.path.splitext(x)[0].split("_")[-1].replace("slice", ""))


def _cmp(a, b):
    aa = "".join(os.path.basename(a).split("_")[:-1])
    bb = "".join(os.path.basename(b).split("_")[:-1])
    if aa < bb: return -1
    if aa > bb: return 1
    return _key(a) - _key(b)


SORT_KEY = cmp_to_key(_cmp)


def main():
    parser = argparse.ArgumentParser(description="Run local EDGE inference from cached Jukebox features")
    parser.add_argument("song_name", help="Folder name under artifacts/cached_features/ or EDGE/cached_features/")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for diffusion sampling")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    song = args.song_name
    feat_dir = os.path.join(EDGE_DIR, "cached_features", song)
    if not os.path.isdir(feat_dir):
        feat_dir = os.path.join(CACHED_FEATURES_DIR, song)
    if not os.path.isdir(feat_dir):
        print(f"No such dir: {feat_dir}")
        sys.exit(1)

    wavs = sorted(glob.glob(os.path.join(feat_dir, "*.wav")), key=SORT_KEY)
    npys = sorted(glob.glob(os.path.join(feat_dir, "*.npy")), key=SORT_KEY)
    assert len(wavs) == len(npys) and len(wavs) > 0, \
        f"Need matched *.wav/*.npy in {feat_dir}"
    print(f"Found {len(wavs)} slices for {song}")

    cond = np.stack([np.load(p) for p in npys])  # (N, 150, 4800)
    print(f"cond shape: {cond.shape}  (slices, frames, jukebox_dim)")
    cond_t = torch.from_numpy(cond).float()

    print("Loading EDGE checkpoint ...")
    model = EDGE("jukebox", os.path.join(EDGE_DIR, "checkpoint.pt"))
    model.eval()
    device = model.accelerator.device
    print("device:", device)

    render_out = os.path.join(EDGE_DIR, "renders", song)
    os.makedirs(render_out, exist_ok=True)
    fk_out = OUT_DIR

    print("Sampling (mode=long, ddim)...")
    data_tuple = (None, cond_t, wavs)
    # render=False so it doesn't call blender / skeleton_render
    model.render_sample(
        data_tuple, song, render_out,
        render_count=-1, fk_out=fk_out, render=False,
    )
    # EDGE saves as <song>_<something>.pkl — find it and normalize the schema.
    cand = sorted(
        glob.glob(os.path.join(fk_out, f"{song}*.pkl")),
        key=os.path.getmtime, reverse=True,
    )
    assert cand, "No pkl written"
    src = cand[0]
    with open(src, "rb") as f:
        d = pickle.load(f)
    # Add smpl_scaling so our load_motion works.
    poses = np.asarray(d["smpl_poses"], dtype=np.float32).copy()
    trans = np.asarray(d["smpl_trans"], dtype=np.float32).copy()
    poses[:, :3] = (Z_UP_TO_Y_UP * R.from_rotvec(poses[:, :3])).as_rotvec()
    trans = Z_UP_TO_Y_UP.apply(trans)

    out = {
        "smpl_poses": poses,
        "smpl_trans": trans,
        "smpl_scaling": np.array([1.0], dtype=np.float32),
    }
    final_path = os.path.join(OUT_DIR, f"{song}.pkl")
    with open(final_path, "wb") as f:
        pickle.dump(out, f)
    print(f"Wrote {final_path}  poses={out['smpl_poses'].shape}  "
          f"trans={out['smpl_trans'].shape}")


if __name__ == "__main__":
    main()
