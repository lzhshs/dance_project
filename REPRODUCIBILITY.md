# Reproducibility Guide

This document records the environment, input artifacts, and commands needed to
reproduce the current final-project results.

## Tested local environment

The current machine reports these package versions:

```text
numpy==2.2.6
scipy==1.15.3
mujoco==3.6.0
imageio==2.37.3
imageio-ffmpeg==0.6.0
torch==2.11.0
librosa==0.11.0
matplotlib==3.10.7
tqdm==4.67.1
```

`requirements.txt` uses compatible lower bounds rather than strict pins. If you
need bit-for-bit comparison, install the exact versions above where possible.

## Input artifact checklist

For the included `pop` reproduction:

```text
generated_motions/pop.pkl
optimized_motions/pop_optimized.npz
optimized_motions/pop_balance_safe.npz
optimized_motions/pop_feasible.npz
optimized_motions/pop_support_com.npz
pop.wav
mujoco_menagerie/unitree_g1/scene.xml
```

For end-to-end generation from audio features, also prepare:

```text
EDGE/checkpoint.pt
cached_features/pop/pop_slice0.wav
cached_features/pop/pop_slice0.npy
cached_features/pop/pop_slice1.wav
cached_features/pop/pop_slice1.npy
cached_features/pop/pop_slice2.wav
cached_features/pop/pop_slice2.npy
```

`edge_infer.py` also accepts the feature folder at `EDGE/cached_features/pop/`.

## Exact final evaluation command

```bash
python final_evaluate.py --motion generated_motions/pop.pkl --wav pop.wav --fps 30 --max-seconds 10
```

This writes:

```text
reports/final/final_metrics.csv
reports/final/final_metrics_summary.md
reports/final/figures/final_stability_curve.pdf
reports/final/figures/final_stability_curve.png
```

## Expected final metrics

Current expected values from `reports/final/final_metrics.csv`:

| Method | Stability | Fall time | Mean joint error | Pelvis min | Motion amp | Beat align |
|---|---:|---:|---:|---:|---:|---:|
| Direct retarget | 11.8% | 1.178 s | 0.0681 rad | 0.084 m | 0.1412 | 0.6866 |
| Offline-filtered | 16.3% | 1.630 s | 0.0132 rad | 0.058 m | 0.0798 | n/a |
| Balance-safe | 100.0% | none | 0.0034 rad | 0.789 m | 0.0143 | n/a |
| Feasibility-optimized | 100.0% | none | 0.0065 rad | 0.788 m | 0.0516 | n/a |
| Support/COM-aware | 100.0% | none | 0.0076 rad | 0.788 m | 0.0606 | n/a |

Small numeric drift is expected across MuJoCo/Python versions. The qualitative
check is that the first two rows fall early and the final three remain stable for
10 seconds.

## Rebuild artifacts from `generated_motions/pop.pkl`

Use this when you want to regenerate `.npz` references before evaluation:

```bash
python optimize_g1_motion.py generated_motions/pop.pkl --fps 30 --out optimized_motions/pop_optimized.npz
python make_balance_safe.py optimized_motions/pop_optimized.npz --out optimized_motions/pop_balance_safe.npz
python robot_feasibility_optimize.py optimized_motions/pop_optimized.npz --out optimized_motions/pop_feasible.npz --samples 90 --seed 7 --max-seconds 10
python support_com_optimize.py optimized_motions/pop_optimized.npz --out optimized_motions/pop_support_com.npz --samples 160 --seed 11 --max-seconds 10
python final_evaluate.py --motion generated_motions/pop.pkl --wav pop.wav --fps 30 --max-seconds 10
```

## Render videos

```bash
python pd_track.py generated_motions/pop.pkl --fps 30 --pin-root
python pd_track.py optimized_motions/pop_optimized.npz
python pd_track.py optimized_motions/pop_balance_safe.npz
python pd_track.py optimized_motions/pop_feasible.npz
python pd_track.py optimized_motions/pop_support_com.npz
python add_audio.py videos/pop_support_com_pd.mp4 pop.wav
```

If rendering fails on a headless machine, run with an explicit MuJoCo GL backend,
for example:

```bash
MUJOCO_GL=egl python pd_track.py optimized_motions/pop_support_com.npz
```

## Full regeneration from cached Jukebox features

```bash
python edge_infer.py pop --seed 7
python optimize_g1_motion.py generated_motions/pop.pkl --fps 30
python make_balance_safe.py optimized_motions/pop_optimized.npz
python robot_feasibility_optimize.py optimized_motions/pop_optimized.npz --samples 90 --seed 7 --max-seconds 10
python support_com_optimize.py optimized_motions/pop_optimized.npz --samples 160 --seed 11 --max-seconds 10
python final_evaluate.py --motion generated_motions/pop.pkl --wav pop.wav --fps 30 --max-seconds 10
```

EDGE diffusion sampling can still vary slightly between hardware backends. For
paper/report reproduction, prefer the checked `generated_motions/pop.pkl`.
