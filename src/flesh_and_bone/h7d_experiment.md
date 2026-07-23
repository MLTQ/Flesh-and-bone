# `h7d_experiment.py`

## Intent

This follow-up distinguishes ecological stability from causal excitation without
retraining H7C. It imports the canonical H7C configurations and checkpoints,
constructs the predeclared 2x Kimodo resampling, and applies the full original
H7 gate set to 20-cycle rollouts.

## Contracts

- The source H7C aggregate must already exist; no model parameters are updated.
- `periodic_catmull_rom` resamples 58 phases to 29 while retaining phase zero.
  The runner validates that phase count against the frozen 2x multiplier.
- Untouched Kimodo stability is read from immutable H7C metrics. Only the three
  causal/non-vacuity gates are excluded from that arm's verdict.
- The 2x arm must pass the complete `acceptance_h7` result for every seed.
- Training NRMSE remains the declared fitted-model gate; 2x one-step NRMSE is
  reported as a non-gating extrapolation diagnostic, matching H7C final policy.
- Final-cycle tensors are retained for rendering without rerunning mechanics.

H7D reuses an already opened motion and therefore cannot count as independent
semantic generalization. Its claim is narrower: whether controlled temporal
excitation makes the same bounded density mechanism necessary and stable.
