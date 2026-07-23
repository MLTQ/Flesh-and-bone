# `render_h7_comparison.py`

## Purpose

Creates matched visual evidence for H7C's severe fast pass and H7D's frozen
Kimodo 2x pass. Each animation places the density-blind H6C backbone beside the
bounded hybrid using identical phase, camera, skeleton, texture, splat settings,
and a seed-7 final-cycle state.

## Outputs

For `fast_1p526` and `kimodo_2x`, the script writes:

- `backbone_vs_hybrid.gif`, a textured side-by-side animation;
- `error_heatmap.gif`, where both panels share the backbone's p99 error as the
  blue-to-red ceiling; and
- five-milestone vertical contact sheets for both views.

## Rendering contract

Mechanics are full 91,979-cell 20-cycle rollouts. Rendering deterministically
selects every third cell and increases radius to `0.65 * pitch` for tractable
evidence generation; this affects appearance only, never metrics. The backbone
is recomputed from the frozen rule, while hybrid final-cycle tensors are loaded
from immutable H7C/H7D artifacts. Error colors compare residual state against
the nonlinear teacher and use the same scale on both panels.

The visualizer defaults to 420 px panels and seed 7. These artifacts help locate
errors; quantitative acceptance remains defined only by the experiment JSON.
