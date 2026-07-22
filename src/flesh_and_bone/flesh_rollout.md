# flesh_rollout.py

## Purpose

Builds H5's phase-held-out teacher dataset, trains one normalized shared rule,
and performs free multi-cycle autoregressive rollout with an optional
neighbor-message ablation.

## Components

### `FleshTrainingConfig`
- **Does**: Records optimizer/model budget and rollout cycle count.
- **Default**: Uses 2,400 optimizer steps. The original 1,200-step budget left
  a rare degree-2 soft-surface cell above H5's maximum-error gate; doubling
  exposure corrected that tail on the permitted seed-7 tuning run without
  changing the teacher, features, topology, or acceptance thresholds.

### `teacher_dataset`
- **Does**: Flattens all captured phase/substep/cell states into 17-channel
  features, acceleration targets, and phase labels.

### `train_flesh_rule`
- **Does**: Holds out every fifth phase, fits normalization on training phases,
  samples deterministic batches, and reports held-out acceleration NRMSE.

### `rollout_flesh_rule`
- **Does**: Starts from the teacher's phase-zero state and integrates only MLP
  accelerations for repeated cycles. The neighbor-blind arm zeros both graph
  message vectors without changing weights or other state.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| H5 experiment | Output shaped `(cycle, phase, cell, 3)` | Phase/cycle layout |
| H5 acceptance | Holdout is phase-based, not random examples from same phase | Split semantics |
| Checkpoints | Model includes fitted normalization buffers | State dict changes |
