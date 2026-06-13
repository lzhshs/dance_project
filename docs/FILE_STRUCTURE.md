# File Structure

The repository is organized around code, artifacts, external dependencies, and
presentation/report deliverables.

```text
.
├── g1_dance/                 # Project package: retargeting, optimization, rollout, evaluation
├── artifacts/                # Local data and generated outputs
│   ├── cached_features/      # Jukebox feature slices for EDGE
│   ├── generated_motions/    # EDGE/SMPL .pkl outputs
│   ├── inputs/               # Small input media such as pop.wav
│   ├── optimized_motions/    # G1 .npz references and optimizer outputs
│   └── videos/               # Rendered MuJoCo videos (ignored by git)
├── deliverables/             # Final presentation slides, sheets, and selected videos
├── reports/                  # Midterm/final report assets and generated metrics
├── docs/                     # Project-level documentation and proposal archive
├── tests/                    # Unit tests for command parsers and trajectory utilities
├── vendor/                   # Lightweight local shims for optional upstream packages
├── EDGE/                     # EDGE upstream/submodule code
├── Bailando/                 # Bailando upstream/submodule code
└── mujoco_menagerie/         # MuJoCo robot models; Unitree G1 is used here
```

Run project modules with `python -m g1_dance.<module>` from the repository root.
For example:

```bash
python -m g1_dance.final_evaluate --max-seconds 10
python -m g1_dance.pd_track artifacts/optimized_motions/pop_feasible.npz
```

Cleanup performed during the restructuring:

- Removed Python bytecode caches and platform metadata files.
- Removed LaTeX auxiliary build products from `reports/midterm/`.
- Removed obsolete one-off debug outputs (`debug_frame.py`, `debug_frame.png`, `g1_dance.mp4`).
- Moved local artifacts from the root into `artifacts/`.
- Moved compatibility shims from the root into `vendor/`.
