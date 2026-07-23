# `density_rollout.py`

## Intent

This is the recurrent test of H7. It recomputes graph messages, LBS-relative
density observations, bounded coefficients, and acceleration from the model's
own residual and velocity at every substep. Teacher observations are used only
to initialize phase zero and to supply the known skeleton trajectory.

## Contracts

- The H6C backbone is always active.
- `density_enabled=False` is the backbone-only/density-blind causal control; it
  zeroes the entire new residual while leaving initialization and integration
  unchanged.
- Output stores one residual state per visible phase for every repeated cycle,
  matching `h5_metrics.measure_rollout`.
- Diagnostics independently accumulate RMS, maximum, near-cap fraction, and
  finite-state status for the predicted density term.
- A rollout defaults to 20 cycles for H7, but tests may use shorter runs.

The skeleton/LBS phase sequence repeats. H7 therefore evaluates stability under
periodic forcing; it does not yet address cold starts or nonperiodic streaming.
