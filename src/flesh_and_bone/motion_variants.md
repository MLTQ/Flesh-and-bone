# motion_variants.py

## Purpose

Constructs the deterministic, predeclared H6M temporal-forcing controls from
the original walk without changing the body, skin weights, or material.

## Components

### `MotionCycle`
- **Does**: Couples one named LBS-position cycle to bone endpoints transformed
  by the exact same phase operation.

### `periodic_catmull_rom`
- **Does**: Resamples a closed tensor-valued trajectory to a requested frame
  count with periodic Catmull-Rom interpolation.
- **Rationale**: Linear interpolation produces acceleration impulses at source
  knots and would confound rule generalization with a resampling artifact.

### `reverse_cycle` / `walk_then_hold`
- **Does**: Preserve phase zero while reversing direction, or append a strict
  phase-zero dwell after one traversal.

### `controlled_motion_cycles`
- **Does**: Returns the replay calibration plus reverse, half-speed,
  `29/19x`, and walk-then-hold cycles in the ledger's frozen order.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| H6M runner | Names, order, and frame counts match `experiments/H6M.md` | Transform/default changes |
| Renderer | LBS positions and bone endpoints share phase count | Transforming only one tensor |
| Teacher | At least three uniformly timed phases | Degenerate output cycle |
