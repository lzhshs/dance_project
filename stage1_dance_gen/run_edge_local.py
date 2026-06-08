"""
Local EDGE inference on RTX 5060 — no Jukebox / no pytorch3d wheel needed.

Strategy:
  - Use the pretrained EDGE checkpoint (jukebox variant, 4800-dim cond)
  - Pass dummy zeros as audio conditioning
  - Set guidance_weight = 0  →  pure unconditional dance generation
  - The model still produces high-quality dance from its learned distribution
  - Music–dance alignment is handled downstream by beat-alignment post-processing

Usage:
  conda activate edge
  cd <project>/external/EDGE
  python ../../stage1_dance_gen/run_edge_local.py
"""

import glob
import os
import pickle
import sys
from functools import cmp_to_key
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# ---- paths ---------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
EDGE_DIR = os.path.join(PROJECT_DIR, "external", "EDGE")
MUSIC_DIR = os.path.join(PROJECT_DIR, "music")
CHECKPOINT = os.path.join(EDGE_DIR, "checkpoint.pt")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "generated_motions")

# ensure EDGE is on the path (so its internal imports work)
sys.path.insert(0, EDGE_DIR)

# ---- stub out heavy/unnecessary deps before importing EDGE ---------------
import types
import importlib.machinery

# wandb — only used for training logging
wandb_stub = types.ModuleType("wandb")
wandb_stub.__spec__ = importlib.machinery.ModuleSpec("wandb", None)
wandb_stub.init = lambda **kw: None
wandb_stub.log = lambda *a, **kw: None
class _FakeRun:
    def finish(self): pass
wandb_stub.run = _FakeRun()
sys.modules["wandb"] = wandb_stub

# p_tqdm — only used for parallel rendering (we set render=False)
def _p_map(fn, iterable, **kw):
    return [fn(x) for x in iterable]
ptqdm_stub = types.ModuleType("p_tqdm")
ptqdm_stub.p_map = _p_map
sys.modules["p_tqdm"] = ptqdm_stub

# patch torch.load to accept old-style checkpoints (contains custom Normalizer)
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# ---- import EDGE modules -------------------------------------------------
from EDGE import EDGE
from data.slice import slice_audio
from data.audio_extraction.baseline_features import extract as baseline_extract

# ---- helpers from EDGE test.py -------------------------------------------
key_func = lambda x: int(os.path.splitext(x)[0].split("_")[-1].split("slice")[-1])

def stringintcmp_(a, b):
    aa = "_".join(a.split("_")[:-1])
    bb = "_".join(b.split("_")[:-1])
    ka, kb = key_func(a), key_func(b)
    if aa < bb: return -1
    if aa > bb: return 1
    if ka < kb: return -1
    if ka > kb: return 1
    return 0

stringintkey = cmp_to_key(stringintcmp_)


# ---- conversion helpers --------------------------------------------------
def axis_angle_to_rot6d(aa):
    """(N, 3) axis-angle → (N, 6) 6D rotation."""
    R = Rotation.from_rotvec(aa).as_matrix()
    return np.concatenate([R[:, :, 0], R[:, :, 1]], axis=-1)


def convert_edge_output_to_151(smpl_poses, smpl_trans):
    """
    Convert EDGE pkl output to our pipeline's (K, 151) format.
      smpl_poses: (K, 72) axis-angle for 24 joints
      smpl_trans: (K, 3)  root translation
    Returns: (K, 151) = 144 rot6d + 3 trans + 4 foot_contact
    """
    K = smpl_poses.shape[0]
    joint_aa = smpl_poses.reshape(K, 24, 3)

    joint_rot6d = np.zeros((K, 24, 6))
    for t in range(K):
        joint_rot6d[t] = axis_angle_to_rot6d(joint_aa[t])

    joint_rot6d_flat = joint_rot6d.reshape(K, 144)
    root_trans = smpl_trans

    # foot contact heuristic (ankle velocity thresholding)
    l_vel = np.linalg.norm(np.diff(joint_aa[:, 7], axis=0), axis=-1)
    r_vel = np.linalg.norm(np.diff(joint_aa[:, 8], axis=0), axis=-1)
    l_vel = np.concatenate([[0], l_vel])
    r_vel = np.concatenate([[0], r_vel])
    l_contact = (l_vel < np.median(l_vel)).astype(float)
    r_contact = (r_vel < np.median(r_vel)).astype(float)
    foot_contact = np.stack([l_contact, l_contact, r_contact, r_contact], axis=-1)

    motion = np.concatenate([joint_rot6d_flat, root_trans, foot_contact], axis=-1)
    assert motion.shape == (K, 151), f"Expected (K,151), got {motion.shape}"
    return motion


# ---- main -----------------------------------------------------------------
def generate_for_music(wav_path, genre, model, out_length=10.0):
    """Slice audio → build dummy conditioning → run EDGE diffusion → save."""
    print(f"\n{'='*60}")
    print(f"  Generating dance for: {genre}  ({wav_path})")
    print(f"{'='*60}")

    # 1. Slice audio into 5-second chunks with 2.5 s stride
    slice_dir = os.path.join(EDGE_DIR, "cached_slices", genre)
    os.makedirs(slice_dir, exist_ok=True)
    print("  Slicing audio...")
    slice_audio(wav_path, 2.5, 5.0, slice_dir)
    wav_slices = sorted(glob.glob(os.path.join(slice_dir, "*.wav")), key=stringintkey)
    print(f"  Got {len(wav_slices)} slices")

    # 2. Determine how many slices we need
    sample_size = int(out_length / 2.5) - 1
    n_use = min(len(wav_slices), sample_size)
    if n_use < 1:
        n_use = 1
    file_list = wav_slices[:n_use]
    print(f"  Using {n_use} slices for ~{out_length}s output")

    # 3. Build dummy conditioning of shape (n_use, 150, 4800)
    #    The model expects jukebox features but we set guidance_weight=0,
    #    so only the unconditional path matters.  We still need the right shape.
    cond = torch.zeros(n_use, 150, 4800)
    print(f"  Conditioning shape: {cond.shape}  (dummy — unconditional mode)")

    # 4. Override guidance weight to 0 (unconditional)
    original_gw = model.diffusion.guidance_weight
    model.diffusion.guidance_weight = 0
    print(f"  guidance_weight = {model.diffusion.guidance_weight}  (unconditional)")

    # 5. Run diffusion
    print("  Running DDIM sampling...")
    data_tuple = (None, cond, file_list)
    fk_dir = os.path.join(EDGE_DIR, "fk_out")
    os.makedirs(fk_dir, exist_ok=True)
    render_dir = os.path.join(EDGE_DIR, "renders")
    os.makedirs(render_dir, exist_ok=True)

    # Monkey-patch EDGE.render_sample to pass sound=False
    # (avoids soundfile crash on Chinese paths, and we only need motion data)
    _orig_diff_render = model.diffusion.render_sample
    def _patched_render(*args, **kwargs):
        kwargs["sound"] = False
        kwargs["render"] = False
        return _orig_diff_render(*args, **kwargs)
    model.diffusion.render_sample = _patched_render

    model.render_sample(
        data_tuple,
        label=genre,
        render_dir=render_dir,
        render_count=-1,
        fk_out=fk_dir,
        render=False,
    )

    # restore
    model.diffusion.render_sample = _orig_diff_render

    # restore guidance weight
    model.diffusion.guidance_weight = original_gw

    # 6. Find the output pkl and convert
    pkl_files = sorted(glob.glob(os.path.join(fk_dir, f"{genre}*.pkl")))
    if not pkl_files:
        pkl_files = sorted(glob.glob(os.path.join(fk_dir, "*.pkl")))

    if not pkl_files:
        print("  ERROR: no pkl output found!")
        return None

    pkl_path = pkl_files[-1]  # latest
    data = pickle.load(open(pkl_path, "rb"))
    smpl_poses = data["smpl_poses"]  # (K, 72)
    smpl_trans = data["smpl_trans"]  # (K, 3)
    print(f"  EDGE output: poses {smpl_poses.shape}, trans {smpl_trans.shape}")

    # 7. Convert to (K, 151) and save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    motion_151 = convert_edge_output_to_151(smpl_poses, smpl_trans)
    npy_path = os.path.join(OUTPUT_DIR, f"{genre}_motion.npy")
    np.save(npy_path, motion_151)
    print(f"  Saved: {npy_path}  shape={motion_151.shape}")

    # also keep the raw pkl
    raw_path = os.path.join(OUTPUT_DIR, f"{genre}_raw.pkl")
    pickle.dump(data, open(raw_path, "wb"))
    print(f"  Saved: {raw_path}")

    return npy_path


def main():
    print("="*60)
    print("  EDGE Local Inference — RTX 5060 (unconditional mode)")
    print("="*60)

    # check checkpoint
    if not os.path.exists(CHECKPOINT):
        print(f"ERROR: Checkpoint not found at {CHECKPOINT}")
        print("Download from: https://drive.google.com/file/d/1BAR712cVEqB8GR37fcEihRV_xOC-fZrZ")
        sys.exit(1)

    # check music
    genres = []
    for name in ["ballet", "hiphop"]:
        wav = os.path.join(MUSIC_DIR, f"{name}.wav")
        if os.path.exists(wav):
            genres.append((name, wav))
        else:
            print(f"WARNING: {wav} not found, skipping {name}")

    # also try house if exists
    house_wav = os.path.join(MUSIC_DIR, "house.wav")
    if os.path.exists(house_wav):
        genres.append(("house", house_wav))

    if not genres:
        print("ERROR: No music files found in", MUSIC_DIR)
        sys.exit(1)

    print(f"\nMusic files: {[g[0] for g in genres]}")
    print(f"Checkpoint:  {CHECKPOINT} ({os.path.getsize(CHECKPOINT)/1e6:.0f} MB)")
    print(f"Output dir:  {OUTPUT_DIR}")

    # load model
    print("\nLoading EDGE model...")
    os.chdir(EDGE_DIR)  # EDGE expects to run from its own directory
    model = EDGE("jukebox", CHECKPOINT)
    model.eval()
    print("Model loaded!")

    # generate for each genre
    for genre, wav_path in genres:
        generate_for_music(wav_path, genre, model, out_length=10.0)

    # summary
    print("\n" + "="*60)
    print("  DONE — Generated motions:")
    print("="*60)
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fp = os.path.join(OUTPUT_DIR, f)
        print(f"  {f}: {os.path.getsize(fp)/1024:.0f} KB")

    print(f"\nNext steps:")
    print(f"  cd {os.path.join(PROJECT_DIR, 'stage2_retarget')}")
    print(f"  conda activate sim")
    print(f"  python retarget_smplsim.py")


if __name__ == "__main__":
    main()
