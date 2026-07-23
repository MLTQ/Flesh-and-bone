# `density_rule.py`

## Intent

H7 does not let a generic MLP predict acceleration. This module confines
learning to two scalar gains applied to declared rotation-equivariant density
vectors. The known-stable H6C constitutive rule remains a frozen additive
backbone.

## Model contract

The five scalar inputs are signed mean compression, compression RMS, stretch
RMS, normalized bone softness, and residual speed divided by `pitch * fps`.
They are dimensionless and explicitly clipped to finite diagnostic ranges.
The final sigmoid bounds pressure to `[0, 60]` and cohesion to `[0, 18]`.
`smooth_norm_cap` then bounds their combined acceleration at 12 m/s².

The final layer starts at zero logits, giving midpoint coefficients (30 and 9)
rather than a random initial force law. Earlier layers and both logits remain
trainable. The learner cannot choose force direction: the teacher-derived
compression and stretch vectors provide it.

The same feature construction has separate captured-trajectory and live-state
entry points. They share units and clipping, preventing a training/rollout
contract mismatch.

## Data and fitting

`sample_density_states` gathers states directly from a captured nonlinear
trajectory using deterministic random indices. This avoids constructing a full
expanded feature matrix for the roughly 92k-cell volume. Each training motion
contributes the same number of examples before concatenation, preventing its
phase count from silently changing motion weight.

The reported NRMSE uses centered target energy and is evaluated in chunks.
Checkpoints include both the frozen constitutive buffers and bounded residual
parameters through `HybridDensityRule`.
