# H7D — frozen Kimodo excitation audit

## Question

Did H7C's Kimodo arm fail because the bounded rule cannot generalize to the
external motion, or because the gentle clip does not excite enough nonlinear
density mechanics to make the backbone control meaningfully wrong?

## Frozen protocol

H7D performs no training and changes no teacher, feature, parameter, checkpoint,
or gate. It loads the three frozen H7C residuals and evaluates:

1. the untouched 58-phase Kimodo result already opened by H7C as an ecological
   stability arm; and
2. a deterministic 29-phase periodic Catmull-Rom resampling of the same LBS
   cycle, exactly 2x temporal speed, as a causal excitation stress.

The transformation is temporal only. Geometry, cell graph, skinning, phase-zero
origin, H7C teacher coefficients, 12 m/s² cap, and 20-cycle duration are
unchanged. The 2x factor and 29-frame count are frozen before its nonlinear
teacher or any checkpoint rollout is run.

## Predeclared verdicts

The untouched ecological arm passes stability if every original H7 gate except
`nonvacuous_backbone`, `position_causal`, and `compression_causal` passes. Those
three causal values remain reported but cannot turn a low-excitation clip into
evidence for or against the density mechanism.

The 2x stress must pass **every original H7 gate**, including backbone error at
least `0.2 mm`, at least 60% position-error reduction, and at least 50%
compression-error reduction, for all seeds. No retraining is permitted.

H7D passes only if untouched Kimodo passes ecological stability and every frozen
checkpoint passes the full 2x stress. This is a controlled excitation audit of
an already opened motion, not a second independent semantic holdout.

## Status

Complete; aggregate **pass**. Untouched Kimodo passes the ecological stability
verdict. At 2x speed, density acceleration rises to `0.24624 m/s²` RMS and the
backbone miss reaches a non-vacuous `0.47698 mm`. Frozen hybrids remain at
`0.000957–0.001102 mm` RMS over 20 cycles, removing 99.77–99.80% of position
error and 99.77–99.78% of sampled compression error. All absolute, amplitude,
softness, drift, finite-state, causal, and acceleration-cap gates pass for every
seed. Stress one-step NRMSE is `0.00290–0.00343` despite no retraining.

**Decision:** retain the H7C bounded hybrid architecture. H7C already proves
causal benefit on the independent fast holdout; H7D establishes that the
external Kimodo semantics are also compatible once temporally excited. The
untouched clip's H7C failure should be cited as a non-vacuity/protocol failure,
not model instability. H7D is not another semantic holdout because it transforms
an already opened clip.
