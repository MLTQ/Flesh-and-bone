# h6m_experiment.py

## Purpose

Orchestrates the H6M calibration and controlled zero-shot motion trials using
the three immutable H5U checkpoints and 91,979-cell graph.

## Components

### `H6MConfig`
- **Does**: Records checkpoint/asset paths, seeds, ten-cycle horizon,
  deterministic feature-sample budget, and dense render settings.

### `_load_rule`
- **Does**: Reconstructs the 96-channel rule and loads checkpoint weights plus
  normalization buffers with `weights_only=True`.
- **Rationale**: H6M contains no optimizer or normalization-fitting path.

### `_render_motion`
- **Does**: Writes five-phase teacher and frozen-rule contact sheets for each
  novel arm at H5U's accepted splat density and opacity.

### `run_h6m`
- **Does**: Builds each motion-specific converged teacher, evaluates full and
  neighbor-blind ten-cycle rollouts for all seeds, records feature shift, writes
  resumable per-seed/per-motion JSON, and applies the strict aggregate verdict.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| H6M ledger | Replay first, followed by exactly four controlled arms | Motion order/count |
| H5U checkpoints | Plain `FleshResidualRule(96)` state dictionaries | Model/checkpoint format |
| Research audit | Per-seed artifacts written before aggregate result | Delayed-only output |
| Visual evidence | Rendered learned state is the final of ten free cycles | Substituting teacher state |
