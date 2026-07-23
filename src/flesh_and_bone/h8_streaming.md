# `h8_streaming.py`

## Purpose

Provide H8's cold-start, nonperiodic mechanics substrate. It owns finite-time
resampling, final-pose holds, non-wrapping skeleton acceleration, one explicit
teacher pass, and frozen-rule streaming rollouts.

## Components

### `H8MotionStream`

- **Does:** Carries resampled LBS positions, bone endpoints, timing, and the
  boundary between generated motion and the final hold.

### `build_motion_stream` / `nonperiodic_resample`

- **Does:** Linearly time-resamples a finite clip and appends 30 exact final-pose
  frames without wrapping endpoints.
- **Rationale:** H8 must not inherit H6K's truncation or palindrome closure.

### `nonperiodic_acceleration`

- **Does:** Computes the second finite difference with one-sided endpoints.
- **Rationale:** Periodic acceleration would couple the last hold pose back to
  the first generated frame and manufacture an impulse.

### `simulate_streaming_teacher`

- **Does:** Starts explicit H7C mechanics from zero residual/velocity, stores
  visible states, and gathers a deterministic bounded one-step sample.
- **Interacts with:** Density observations in `density_teacher.py` and the H5U
  voxel graph.

### `rollout_streaming_density`

- **Does:** Runs the frozen H6C backbone with density enabled/disabled from the
  same zero state and records safety diagnostics.
- **Interacts with:** Frozen `HybridDensityRule` checkpoints from H7C.

## Contracts

| Dependent | Expects | Breaking changes |
| --- | --- | --- |
| `h8_experiment.py` | visible state shape is `[frame,cell,3]` | Shape/timing semantics |
| `h8_metrics.py` | motion/hold boundary and final velocity are retained | Removing boundary/final fields |
| H8 protocol | no warmup, wrap, repeat, or teacher initialization | Changing initial/boundary conditions |

## Notes

The diagnostic dataset samples the same fixed cell subset at every substep. It
measures frozen-rule extrapolation without retaining every full per-substep
teacher field.
