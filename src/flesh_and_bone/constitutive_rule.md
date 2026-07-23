# constitutive_rule.py

## Purpose

Provides H6C's structure-preserving local rule and deterministic closed-form
system identification from the original H5U teacher states.

## Components

### `constitutive_terms`
- **Does**: Maps the established 17 raw H5 features into five vector terms:
  stiffness restoring force, square-root-stiffness damping, neighbor residual,
  neighbor velocity, and inertial LBS forcing.
- **Rationale**: Coefficients remain scalar and shared across xyz, preserving
  rotation equivariance and linear growth in dynamic state.

### `ConstitutiveFleshRule`
- **Does**: Applies five frozen identified scalars to the basis and presents the
  same `rule(features) -> acceleration` contract as the H5 MLP.

### `fit_constitutive_rule`
- **Does**: Accumulates scaled double-precision 5x5 normal equations over all
  non-holdout phase states, solves once, and reports phase-held-out NRMSE/RMSE.
- **Rationale**: Column scaling avoids conditioning loss between unit-scale
  inertial terms and the roughly 1200x graph coefficient; streaming avoids a
  multi-gigabyte design matrix.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| H5 rollout | Rule accepts raw 17-channel features and returns xyz acceleration | Feature/basis order |
| H6C ledger | Phase modulo five equals four is held out | Split semantics |
| Checkpoint/runtime | Exactly five scalar coefficients, no normalization state | Model format |
