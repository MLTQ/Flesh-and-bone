# h2_metrics.py

## Purpose

Measures H2 region-wise anatomy, wound recovery, replacement-cell lineage,
local maturation, and articulated persistence independently of control loss.

## Components

### `measure_h2`
- **Does**: Extends common H1 evidence with per-region coverage, allocation
  error, wound/healthy coverage, radial profile error, and generation-1
  localization/commitment/lock fractions. It also reports coarse guided-cell
  count so region-level developmental privilege remains visible.
- **Rationale**: Global coverage can hide a missing bulb or pad, just as global
  color balance hid H1 texture collapse.

### `acceptance_h2`
- **Does**: Applies the predeclared development, damage, repair, phenotype,
  lineage, motion, mass, and assignment-leakage gates in `experiments/H2.md`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `h2_experiment.py` | Four phase measurements and stable JSON keys | Signature/key changes |
| Experiment ledger | Thresholds exactly match the predeclared H2 protocol | Gate changes |
