# H6C — structure-preserving constitutive identification

## Question

Is H6M's fast-motion failure caused by insufficient local state, or by using an
unconstrained MLP to approximate a stable constitutive law?

## Model

H6C retains the exact H5U graph, features, material teacher, 91,979 cells, and
single-walk training data, but replaces the free 17→96→96→3 MLP with five
shared coefficients:

```text
a = c0 (-stiffness * residual)
  + c1 (-sqrt(stiffness) * velocity)
  + c2 neighbor_residual
  + c3 neighbor_velocity
  + c4 (-LBS_acceleration)
```

The coefficients are identified once from the non-holdout phases of the
original walk by scaled double-precision normal equations. This is supervised
system identification, not a hand-copy of teacher constants: the fitter sees
the same state/acceleration examples as H5U. Its hypothesis class is deliberately
chosen to preserve linear growth in forcing and damping/stiffness sign.

The graph-elastic teacher lies exactly in this class, so H6C is a structural
upper-bound control. A pass will not establish realistic tissue; it will show
that the H6M failure is regression architecture rather than absent information.

## Protocol

- Fit only on original-walk phases whose index modulo five is not four.
- Report coefficients and held-out acceleration NRMSE; no optimizer or random
  seed is used.
- Freeze the five coefficients.
- Run the identical H6M replay, reverse, half-speed, `1.526x`, and
  walk-then-hold teachers for ten cycles.
- Run the same neighbor-blind ablation and H6M rollout gates.
- Render the final-cycle fast-motion result, where the MLP control visibly
  failed.

## Predeclared acceptance criteria

- Held-out acceleration NRMSE at most `1e-4`.
- Identified nonzero coefficients lie within 1% of the teacher-implied values
  `[1, 0.44, 1200, 1]` for stiffness, damping, neighbor coupling, and inertial
  forcing; neighbor-velocity magnitude is at most `1e-3`.
- Replay and all four novel motions pass every H6M teacher, rollout,
  amplitude, softness, drift, LBS-improvement, finite-state, and
  neighbor-causality gate.

No threshold changes are permitted after fitting.

## Interpretation discipline

Passing favors a hybrid design: a stable, equivariant constitutive backbone
plus a small bounded learned residual for mechanics absent from the backbone.
It does not justify hard-coding this synthetic teacher as the final flesh model.
Failing would mean that even the correct local basis is insufficient under the
current integration, initialization, or graph discretization.

## Results

H6C **passes every predeclared gate**. Scaled normal equations identify:

```text
stiffness          1.00000000005
damping            0.43999999790
neighbor residual 1199.99999510
neighbor velocity  0.00000012614
LBS inertia        1.00000000000
```

Held-out acceleration NRMSE is `5.48e-8`. All replay and novel-motion rollouts
remain within `5.5e-10–4.2e-9 m` RMS and below `9.5e-8 m` maximum after ten
cycles, including the `1.526x` motion that catastrophically destabilized every
MLP. Neighbor blindness is roughly one million times worse because the fitted
full-rule error is at floating-point floor. The full run takes 62.8 seconds.

## Decision

H6M's fast failure is not missing local information, graph resolution, or the
integrator: it is unconstrained function approximation. Retain the five-term
structure as the guaranteed-stable backbone. Future learned capacity should be
a bounded residual for mechanics outside this basis, trained against a richer
teacher with contact, density, or incompressibility; reproducing this linear
teacher with another free MLP is no longer informative.

Canonical evidence is in `experiments/runs/h6c_final`.
