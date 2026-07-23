# H7C — capped non-vacuous density mechanics

## Why H7C exists

H7B's half-speed qualification made density acceleration nontrivial at `0.06051
m/s²` RMS and again passed every accuracy, causal-reduction, and safety gate.
The backbone position miss rose from H7's `0.0327 mm` to `0.1438 mm`, but
remained below the unchanged `0.2 mm` non-vacuity floor. The bounded hybrid
reached `0.00512` unseen one-step NRMSE and removed 99.79% of the backbone error.
Fast and Kimodo holdouts remained sealed.

## Frozen correction

H7C scales the original H7 coefficients by 24x, or H7B by 3x:

- teacher pressure near/far: `720 / 1200`;
- teacher cohesion near/far: `144 / 288`;
- learned pressure maximum: `1440`;
- learned cohesion maximum: `432`.

The scale was frozen from H7B before H7C ran. Only 1.391x more backbone error is
needed to cross the fixed non-vacuity floor; 3x coefficient scale leaves margin
for the already observed sublinear response as rare cells approach the smooth
cap. The teacher and learned density acceleration remain bounded by the same
`12 m/s²` smooth norm cap.

All other teacher definitions, model features and width, optimizer settings,
training examples, data split, seeds, 20-cycle duration, controls, and gates are
unchanged from H7. Failed H7 and H7B artifacts remain immutable.

## Status

Half-speed seed-7 qualification **passes every frozen gate**. Teacher density
acceleration is `0.11215 m/s²` RMS and bounded at `11.9883 m/s²`. The backbone
miss is `0.24619 mm`, clearing the non-vacuity floor. The hybrid reaches
`0.00432` unseen one-step NRMSE, `0.000470 mm` 20-cycle position RMS, and removes
99.81% of backbone position error plus 99.83% of sampled compression error.
Near-cap occupancy is `1.41e-7`, so the result is not produced by broad force
saturation.

The H7C configuration is now frozen. Fast and Kimodo final holdouts may be
opened once for seeds 7, 19, and 31.

Final strict aggregate **fails**, with a clean separation. The `1.526x` fast
holdout passes all seeds: the backbone misses by `4.010 mm` RMS while hybrids
remain at `0.1168–0.1187 mm`, removing 97.04–97.09% of position error and
94.14–94.24% of compression error. Untouched Kimodo is stable and almost exact
(`0.000230–0.000311 mm` hybrid RMS), but fails the fixed non-vacuity gate because
its backbone miss is only `0.09694 mm`. No gate is waived. H7D predeclares a
frozen-checkpoint temporal stress of that same ecological clip in
[`H7D.md`](H7D.md).
