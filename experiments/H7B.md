# H7B — non-vacuous bounded density mechanics

## Why H7B exists

The frozen H7 initial qualification failed two gates while passing every
learning, rollout, causal-reduction, and safety gate. On half speed, density
acceleration was only `0.01437 m/s²` RMS against the declared `0.02` floor, and
the H6C backbone was already within `0.0327 mm` of the nonlinear teacher against
the declared `0.2 mm` non-vacuity floor. The hybrid reached `0.00444` unseen
one-step NRMSE and removed 99.66% of the position error. This is accurate but not
a meaningful stress test.

The failed `experiments/runs/h7_initial` artifacts remain immutable. Fast and
Kimodo final holdouts were not opened.

## Frozen H7B correction

H7B changes only the physical scale and the matching learned coefficient
bounds:

- teacher pressure near/far: `240 / 400` instead of `30 / 50`;
- teacher cohesion near/far: `48 / 96` instead of `6 / 12`;
- learned pressure maximum: `480` instead of `60`;
- learned cohesion maximum: `144` instead of `18`.

This is an exact 8x scale change. The 5% compression and 8% stretch dead zones,
squared excess-strain vectors, five invariant inputs, `5 -> 32 -> 32 -> 2`
network, sigmoid nonnegativity, `12 m/s²` smooth acceleration cap, optimizer,
training examples, motion split, seeds, and all acceptance gates remain
unchanged.

The factor was fixed from the failed qualification before H7B ran: the measured
uncapped regime predicts approximately `0.115 m/s²` teacher force RMS and
`0.262 mm` backbone error. Both clear the non-vacuity floors without approaching
the `4 m/s²` teacher RMS ceiling. Tail behavior remains independently bounded by
the unchanged norm cap.

## Protocol and gates

Training uses replay, reverse, and walk-then-hold. Half speed seed 7 is the only
qualification. Final fast and Kimodo holdouts remain sealed until it passes;
then seeds 7, 19, and 31 must pass both. All teacher, one-step, 20-cycle,
absolute-error, causal-improvement, and safety gates are exactly those in
[`H7.md`](H7.md).

## Status

Qualification **failed one non-vacuity gate**. Density acceleration reached
`0.06051 m/s²` RMS and all learning, rollout, causal, and safety gates passed,
but the density-blind backbone error was `0.1438 mm` against the unchanged `0.2
mm` floor. The hybrid's half-speed one-step NRMSE was `0.00512` and it removed
99.79% of the position error. Final holdouts were not opened. H7C is
predeclared in [`H7C.md`](H7C.md).
