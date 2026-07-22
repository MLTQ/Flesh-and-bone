# render.py

## Purpose

Produces inspection-first front/tilted projections of arbitrary-bone-count
scaffolds and their mobile Gaussian cells, including plastic/locked checker
phenotype, per-material radius scale, and visibly orange uncommitted cells.

## Components

### `render_frame`
- **Does**: Draws skeleton segments, depth-sorts active cells, and alpha-composes
  isotropic Gaussian splats with an explicit base world-space radius multiplied
  by each cell's material scale. Camera extent and the legacy pixel-radius floor
  are configurable; H0–H2 defaults remain unchanged.
- **Rationale**: A fixed camera and color contract make material sliding, gap
  filling, and late differentiation visually comparable across runs.

### `save_gif` / `save_contact_sheet`
- **Does**: Write animation and milestone evidence using Pillow only.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `experiment.py` | PIL images from `render_frame` and unchanged H0–H2 defaults | Return type/default camera |
| H3 runner | Dynamic bone count, wider camera, and sub-H2 splat floor | Frame shape/render arguments |
| Visual review | Orange means uncommitted; navy/cream are checker identities | Color semantics |
