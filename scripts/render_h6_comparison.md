# render_h6_comparison.py

## Purpose

Reconstructs and renders matched full-cycle animations for the exact H6M
`1.526x` seed-7 failure and H6C constitutive success, enabling direct visual
review beyond milestone contact sheets.

## Components

### `_paired_frames`
- **Does**: Places corresponding failure/success frames side by side without
  resizing or changing their cameras.

### `main`
- **Does**: Rebuilds the accepted 91,979-cell fast-motion teacher, free-runs the
  selected frozen H5U MLP and original-walk-fitted constitutive rule for ten
  cycles, renders every phase with identical H5U appearance settings, and
  writes two individual GIFs plus one paired comparison GIF.
- **Rationale**: Rollouts are recomputed from immutable checkpoints rather than
  approximated from five-frame contact sheets. The default 85 ms frame delay is
  intentionally slower than physical 30 Hz playback for inspection.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Visual review | Failure and success share phases, camera, radius, opacity, and texture | Asymmetric rendering |
| H6 evidence | Default seed 7, ten cycles, `fast_1p526`, coupling 1200 | Protocol/default change |
| Artifact viewer | GIFs loop indefinitely and individual frames remain 720 px | Output format/size |
