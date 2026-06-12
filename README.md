# G1 Music-to-Dance Retargeting

This repository reproduces a pipeline that turns music-conditioned SMPL dance
motion into Unitree G1 robot motions in MuJoCo. The workflow combines EDGE for
music-to-SMPL generation, custom SMPL-to-G1 retargeting, offline feasibility
filters, MuJoCo PD tracking, and final quantitative evaluation.

## What is in this repo

- `edge_infer.py`: run EDGE sampling from precomputed Jukebox features and write
  `generated_motions/<song>.pkl`.
- `retarget_smpl_to_g1.py`: baseline Euler retargeting from SMPL to G1 qpos.
- `retarget_v2.py`: improved lower-body retargeting plus IK-style arm mapping.
- `optimize_g1_motion.py`: offline filtering and joint/contact constraints.
- `robot_feasibility_optimize.py`: search for a free-root, dynamically feasible
  reference with higher expressiveness than the conservative fallback.
- `support_com_optimize.py`: final support/COM-aware feasibility optimizer.
- `contact_balance_optimize.py`: experimental contact-aware PD balance
  post-processor for turning a pinned/showcase reference into a more stable
  free-root reference.
- `pd_track.py`: MuJoCo PD rollout and video rendering.
- `evaluate.py` and `final_evaluate.py`: beat alignment, joint-limit, stability,
  tracking, and summary plot/CSV generation.
- `project_paths.py`: all root-relative paths used by the scripts.
- `mujoco_menagerie/unitree_g1/`: Unitree G1 MuJoCo model used by the pipeline.

Generated artifacts are written to:

- `generated_motions/`: EDGE/SMPL `.pkl` motions.
- `optimized_motions/`: optimized G1 `.npz` references.
- `videos/`: MuJoCo renderings and audio-muxed demos.
- `reports/final/`: CSV metrics, plots, and final report assets.

## Setup

Use Python 3.10 or newer. On a fresh checkout:

```bash
git submodule update --init --recursive
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The root repo includes small local compatibility shims for `accelerate`,
`p_tqdm`, and `pytorch3d.transforms` so local EDGE inference can run on machines
where the full upstream packages are not installed.

## External artifacts

Large files are intentionally not tracked in git. To fully reproduce from audio,
prepare these files:

- `EDGE/checkpoint.pt`: EDGE pretrained checkpoint.
- `cached_features/<song>/*.wav` and `cached_features/<song>/*.npy`, or the same
  folder under `EDGE/cached_features/<song>/`: Jukebox features for EDGE. Each
  `.npy` slice should have shape `(150, 4800)`.
- `aist_data/motions/*.pkl`: optional AIST++ SMPL motions for baseline tests.
- `aist_data/music/*.wav`: optional AIST++ music files for audio muxing and beat
  alignment.
- `pop.wav`: music used by the included final-project `pop` example.

If you only want to reproduce the final reported `pop` evaluation from existing
artifacts, the key inputs are `generated_motions/pop.pkl`, `pop.wav`, and the
`.npz` files under `optimized_motions/`.

## Quick reproduction: final `pop` metrics

Run the final evaluation table and stability plot:

```bash
python final_evaluate.py --motion generated_motions/pop.pkl --wav pop.wav --fps 30 --max-seconds 10
```

Expected outputs:

- `reports/final/final_metrics.csv`
- `reports/final/final_metrics_summary.md`
- `reports/final/figures/final_stability_curve.pdf`
- `reports/final/figures/final_stability_curve.png`

The currently checked generated summary reports that direct retargeting falls at
about 1.18 s, while the balance-safe, feasibility-optimized, and support/COM-aware
references remain stable for the 10 s evaluation window.

## Full pipeline from cached audio features

Assume the song is named `pop` and Jukebox feature slices are available under
`cached_features/pop/` or `EDGE/cached_features/pop/`.

```bash
# 1. EDGE audio-conditioned SMPL generation.
python edge_infer.py pop --seed 7

# 2. Optional kinematic visualization of the raw retargeting.
python retarget_v2.py generated_motions/pop.pkl --fps=30

# 3. Offline robot filtering.
python optimize_g1_motion.py generated_motions/pop.pkl --fps 30

# 4. Conservative stable fallback and feasibility-optimized references.
python make_balance_safe.py optimized_motions/pop_optimized.npz
python robot_feasibility_optimize.py optimized_motions/pop_optimized.npz --samples 90 --seed 7 --max-seconds 10
python support_com_optimize.py optimized_motions/pop_optimized.npz --samples 160 --seed 11 --max-seconds 10

# 5. Render PD rollouts.
python pd_track.py optimized_motions/pop_balance_safe.npz
python pd_track.py optimized_motions/pop_feasible.npz
python pd_track.py optimized_motions/pop_support_com.npz

# 6. Add audio to a rendered video.
python add_audio.py videos/pop_support_com_pd.mp4 pop.wav

# 7. Recompute final metrics.
python final_evaluate.py --motion generated_motions/pop.pkl --wav pop.wav --fps 30 --max-seconds 10
```

For a pinned-root showcase video, use:

```bash
python make_showcase_motion.py optimized_motions/pop_optimized.npz --style expressive
python pd_track.py optimized_motions/pop_showcase_expressive.npz --pin-root
python add_audio.py videos/pop_showcase_expressive_pd_pinned.mp4 pop.wav
```

## AIST++ motion baseline

For a raw AIST++ SMPL `.pkl` motion, use 60 fps unless you know otherwise:

```bash
python retarget_v2.py aist_data/motions/gPO_sFM_cAll_d10_mPO1_ch02.pkl --fps=60
python pd_track.py aist_data/motions/gPO_sFM_cAll_d10_mPO1_ch02.pkl --fps 60 --pin-root
python evaluate.py aist_data/motions/gPO_sFM_cAll_d10_mPO1_ch02.pkl --wav aist_data/music/mPO1.wav --fps 60
```

To inspect how much motion is lost during robot retargeting, first export a
direct SMPL-to-G1 baseline and then render it next to the original SMPL motion:

```bash
python export_direct_mapping.py aist_data/motions/gPO_sFM_cAll_d10_mPO1_ch02.pkl --fps 60
python compare_smpl_to_npz.py \
  aist_data/motions/gPO_sFM_cAll_d10_mPO1_ch02.pkl \
  optimized_motions/gPO_sFM_cAll_d10_mPO1_ch02_direct.npz \
  --fps 60
```

`optimize_g1_motion.py` also supports constraint profiles for different
retargeting goals:

```bash
# Conservative default.
python optimize_g1_motion.py generated_motions/pop.pkl --fps 30 --profile safe \
  --out optimized_motions/pop_safe.npz

# Keeps more torso and arm motion for pinned-root demos.
python optimize_g1_motion.py generated_motions/pop.pkl --fps 30 --profile expressive \
  --out optimized_motions/pop_expressive.npz

# Favors free-root stability by strongly damping root/lower-body motion.
python optimize_g1_motion.py generated_motions/pop.pkl --fps 30 --profile balance \
  --out optimized_motions/pop_balance.npz

# Display-first profile: emphasizes upper-body motion while reducing leg noise.
python optimize_g1_motion.py generated_motions/pop.pkl --fps 30 --profile showcase \
  --out optimized_motions/pop_showcase.npz
python pd_track.py optimized_motions/pop_showcase.npz --fps 30 --pin-root
```

Use `--pin-root` only for visualization. It overwrites the floating-base state
with the reference pose so the robot cannot fall; free-root rollouts are the
physical stability check.

### Experimental contact-aware balance correction

`contact_balance_optimize.py` is a post-processing tool for the common case
where an expressive or showcase reference looks good with `--pin-root` but falls
quickly in a free-root MuJoCo rollout. It estimates left/right foot contact
segments, builds a support-center trajectory, damps risky lower-body/root
motion, and applies a clipped PD-style correction to the floating-base `xy`
trajectory.

This tool is meant as a practical display/stability compromise, not a full
physics controller. It can keep more upper-body motion than the conservative
`balance` profile, but aggressive dances may still require strong damping.

Example starting from a display-first `showcase` reference:

```bash
python optimize_g1_motion.py generated_motions/pop.pkl --fps 30 --profile showcase \
  --out optimized_motions/pop_showcase.npz

python contact_balance_optimize.py optimized_motions/pop_showcase.npz \
  --out optimized_motions/pop_contact_balance.npz \
  --root-kp 0.28 --root-kd 0.12 --max-root-step 0.015 \
  --root-motion-gain 0.10 --leg-gain 0.20 --waist-gain 0.15 \
  --arm-gain 0.45 --root-z-gain 0.15 --root-roll-pitch-gain 0.03

python pd_track.py optimized_motions/pop_contact_balance.npz --fps 30
python add_audio.py videos/pop_contact_balance_pd.mp4 pop.wav
```

For motions with obvious foot sliding inside a contact interval, add
`--stable-support`. This uses one median support point per contact segment and
usually reduces root drift:

```bash
python contact_balance_optimize.py optimized_motions/waack_showcase.npz \
  --out optimized_motions/waack_contact_balance.npz \
  --root-kp 0.32 --root-kd 0.14 --max-root-step 0.012 \
  --root-motion-gain 0.055 --leg-gain 0.11 --waist-gain 0.085 \
  --arm-gain 0.30 --root-z-gain 0.12 --root-roll-pitch-gain 0.02 \
  --stable-support
```

Useful tuning rules:

- Increase `--arm-gain` to keep more visible dance motion.
- Decrease `--leg-gain`, `--waist-gain`, and `--root-motion-gain` when the
  free-root rollout falls early.
- Decrease `--max-root-step` when root movement looks jittery.
- Use `--stable-support` when foot contact estimates slide with the source
  motion.

## Reproducibility notes

- Paths are root-relative through `project_paths.py`; the repo no longer depends
  on a user-specific absolute path.
- EDGE generation is stochastic; use `--seed` for repeatable sampling. Existing
  `.pkl` and `.npz` artifacts are the most reliable way to reproduce the exact
  final report numbers.
- MuJoCo free-root stability can vary slightly across MuJoCo, Python, and BLAS
  versions. Compare stability rates and fall times with a small tolerance.
- Rendering requires a working OpenGL context. On headless Linux, set an
  appropriate MuJoCo backend such as `MUJOCO_GL=egl` before running render-heavy
  scripts.

See `REPRODUCIBILITY.md` for an exact command log and expected metrics.
