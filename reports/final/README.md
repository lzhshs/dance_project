# Final Report Package

This folder contains the cleaned final deliverables for the music-to-Unitree-G1 dance retargeting project.

## Paper

- `final_report.pdf` — final 5-page report.
- `final_report.tex` — LaTeX source.
- `references.bib` — bibliography database.
- `IEEEtran.cls` — local conference-style class used for compilation.

To rebuild the PDF from this folder:

```bash
latexmk -pdf -interaction=nonstopmode final_report.tex
```

## Figures

- `figures/pipeline_overview.pdf` / `.png` — full method pipeline.
- `figures/final_stability_curve.pdf` / `.png` — pop-style pelvis-height stability comparison.
- `figures/cross_style_transfer.pdf` / `.png` — cross-style amplitude and direct-fall comparison.

## Data

- `data/final_metrics.csv` — full numeric metrics for the main pop-style comparison.
- `data/final_metrics_summary.md` — compact result table for quick reading.
- `data/supplemental_style_metrics.csv` — Jazz/Ballet-Jazz and Break supplemental quantitative metrics. These are separate from the curated Waack/Pop presentation videos.
- `data/jazz_metrics.csv`, `data/break_metrics.csv` — per-style metric exports.

## Curated Presentation Videos

These six videos are copied from `deliverables/presentation_videos/` because they look better for an oral demo than the raw diagnostic videos. They are qualitative presentation assets, not the same set as the Jazz/Ballet-Jazz and Break supplemental metric exports. The selection logic is to show the stability--expressiveness trade-off with matched pairs: a pinned-root showcase preserves dance expression but is nonphysical, while a free-root stable rollout supports the simulation-stability claim but is more conservative. The failure example explains why this distinction is necessary.

| File | Condition | What it demonstrates | Claim |
|---|---|---|---|
| `videos/01_waack_pinned_showcase_best_visual.mp4` | Waack, pinned-root showcase | Most expressive opening demo | Visual reference only; root is pinned |
| `videos/02_waack_freeroot_stable_physics.mp4` | Waack, free-root stable rollout | Same style after stability-oriented processing | Simulation-stable demo, more conservative motion |
| `videos/03_pop_pinned_showcase_best_visual.mp4` | Pop, pinned-root showcase | Local generated pop motion with stronger visual expression | Visual reference only; root is pinned |
| `videos/04_pop_freeroot_stable_physics.mp4` | Pop, free-root stable rollout | Pop counterpart used to connect demo to main quantitative result | Simulation-stable demo under selected MuJoCo + PD setup |
| `videos/05_pop_expressive_freeroot_fall_example_no_audio.mp4` | Pop, expressive free-root rollout | Failure case when expression is kept too aggressively | Shows why free-root evaluation matters |
| `videos/06_waack_original_showcase_pinned_reference.mp4` | Waack, pinned-root reference backup | Additional visual reference for the source/showcase style | Nonphysical backup visual only |

Recommended presentation order: open with video 01 for visual impact, contrast it with video 02, then use videos 03 and 04 to show the same two-version logic on the pop example. Video 05 is the short motivation/failure clip. Video 06 is a backup if another Waack visual reference is needed.

## Key Takeaway

Directly retargeted music-generated human dance falls quickly in free-root G1 simulation. Physics-aware reference optimization, especially the support/COM-aware selection stage, keeps the rollout stable under the chosen MuJoCo model and PD controller while preserving substantially more dance motion than the conservative balance-safe baseline. For presentation, pinned-root videos should always be labeled as nonphysical visual references, not as evidence of physical stability.
