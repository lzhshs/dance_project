# Two-Version Dance Outputs

This folder contains two output styles for each selected music/dance example.

## Version A: pinned showcase

Use this version when the goal is visual presentation.

- The floating root is pinned during MuJoCo rollout.
- The robot cannot fall because the root follows the reference pose.
- This keeps more expressive arm and body motion.
- This should be described as a visual-only showcase, not as proof of physical
  balance.

Files:

- `pop_pinned_showcase_audio.mp4`
- `waack_pinned_showcase_audio.mp4`

## Version B: free-root stable

Use this version when the goal is physical plausibility.

- The floating root is not pinned.
- The robot is simulated with normal free-root dynamics.
- Contact-aware balance correction and support-foot locking are applied.
- The motion is more conservative, but it demonstrates stable rollout.

Files:

- `pop_freeroot_stable_audio.mp4`
- `waack_freeroot_stable_audio.mp4`

## Comparison sheets

- `pop_comparison_sheet.jpg`
- `waack_comparison_sheet.jpg`

## Recommended wording

For reports or presentations, describe the two outputs as:

> We provide a pinned-root showcase version to preserve dance expressiveness and
> a free-root stable version to evaluate physically plausible robot execution.
> The pinned-root video is for visual comparison only, while the free-root video
> is the stability-oriented result.
