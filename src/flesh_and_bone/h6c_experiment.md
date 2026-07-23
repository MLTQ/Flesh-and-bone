# h6c_experiment.py

## Purpose

Fits H6C's deterministic five-coefficient constitutive rule on the original
walk and evaluates it under the unchanged H6M ten-cycle protocol.

## Components

### `H6CConfig`
- **Does**: Records the accepted H5U asset/volume, ten-cycle horizon, GPU, and
  dense visual settings.

### `_coefficient_acceptance`
- **Does**: Applies the predeclared held-out NRMSE, 1% nonzero-coefficient, and
  near-zero neighbor-velocity gates against teacher-implied values.

### `_render_fast`
- **Does**: Renders teacher and final-cycle fitted-rule contact sheets for the
  exact fast arm that visibly destabilized every H5U MLP checkpoint.

### `run_h6c`
- **Does**: Fits once on replay teacher states, freezes the rule, evaluates full
  and neighbor-blind rollouts on all H6M motions, writes per-motion evidence,
  and combines coefficient plus motion gates.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| H6C ledger | Fit precedes and is frozen for every motion | Per-motion refit |
| H6M comparison | Same motion transforms, teacher, cycles, and gates | Protocol drift |
| Visual audit | Fast render uses the tenth predicted cycle | Earlier-cycle substitution |
