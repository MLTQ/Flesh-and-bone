# h5_experiment.py

## Purpose

Runs H5's separately taught graph-elastic curriculum across frozen training
seeds, evaluates free and neighbor-blind rollouts, checkpoints each local rule,
and renders actual plus explicitly magnified secondary motion.

## Components

### `H5Config`
- **Does**: Records H4 inputs, frozen seeds/device, and which seed receives
  visual artifacts.
- **Contract**: `image_size` controls every normal and exaggerated render; the
  renderer does not silently substitute a fixed resolution.

### `run_h5`
- **Does**: Builds one deterministic teacher/graph, trains each seed, rolls the
  rule freely for three cycles, runs the message ablation, applies gates, writes
  checkpoints/metrics, and aggregates the verdict.

### `_render_seed`
- **Does**: Writes actual teacher/learned GIFs and contact sheets plus separately
  labeled `4x` residual sheets for perceptual inspection.
- **Rationale**: Two-centimeter secondary motion can be difficult to see at
  full-body scale; magnification is useful only when it is impossible to mistake
  for quantitative evidence.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `scripts/run_h5.py` | `run_h5(path, config)` returns aggregate report | Signature/config |
| Experiment ledger | All three seeds must pass; one passing checkpoint is insufficient | Aggregate semantics |
| Future runtime | Checkpoint includes MLP and normalization buffers | State dict changes |
