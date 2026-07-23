# h5_metrics.py

## Purpose

Measures H5 teacher validity, free-rollout error/stability, neighbor-message
causality, and improvement over the zero-residual LBS control.

## Components

### `measure_teacher`
- **Does**: Reports graph connectivity/degree, cycle seam, residual amplitude,
  far/near softness ratio, edge coherence, and finite state.

### `measure_rollout`
- **Does**: Compares repeated predicted cycles to one periodic teacher cycle,
  including RMS/p99/max, amplitude, softness, phase-zero drift, sparse edge
  strain error, LBS improvement, and finite state.

### `flat_quantile`
- **Does**: Computes an exact linearly interpolated flattened quantile from
  order statistics.
- **Rationale**: Torch's ordinary quantile rejects tensors above 2^24 values;
  H6M's ten-cycle 91,979-cell rollout has 26.7 million distances.

### `acceptance_h5`
- **Does**: Applies every threshold frozen in `experiments/H5.md`; the neighbor
  control may establish causality through position or edge-strain degradation.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| H5 experiment | Stable named teacher/rollout/gate keys | Key changes |
| Research ledger | LBS baseline error equals teacher residual amplitude | Baseline semantics |
| H6M metrics | p99 remains exact above Torch's ordinary quantile size limit | Long-horizon scale |
