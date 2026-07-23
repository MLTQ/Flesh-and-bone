# h6m_metrics.py

## Purpose

Measures how far each novel teacher moves outside a frozen H5U checkpoint's
normalization support and applies H6M's predeclared long-rollout gates.

## Components

### `measure_frozen_feature_shift`
- **Does**: Deterministically samples phase/substep/cell states without
  materializing the full 17-channel dataset, then reports absolute z-score
  tails under the checkpoint's original normalization.
- **Rationale**: The 91,979-cell half-speed teacher would otherwise allocate a
  second multi-gigabyte flattened dataset just for diagnostics.

### `acceptance_h6m`
- **Does**: Applies the ledger's teacher validity bounds and unchanged H5
  rollout/amplitude/softness/drift/LBS/neighbor-causality gates.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| H6M runner | Stable metric and gate names | Key/threshold changes |
| Frozen checkpoints | Feature order is exactly `flesh_rule.flesh_features` | Feature reordering |
| Research ledger | Feature shift is diagnostic and never refits normalization | Mutating checkpoint buffers |
