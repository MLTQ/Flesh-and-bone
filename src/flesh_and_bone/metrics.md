# metrics.py

## Purpose

Defines H0/H1 evidence independently of control loss: coverage, tracking,
density, crowding, differentiation balance, spatial checker reconstruction,
negative space, motion retention, assignment leakage, material lock, mass
accounting, and numerical stability.

## Components

### `measure_state`
- **Does**: Computes per-state geometry and density measurements from active
  particles and the current deformed body plan. Unassigned H1 cells are measured
  against their nearest field sample without storing that index.
- **Checker evidence**: Reports both global color balance and covered-site
  checker accuracy. The latter detects spatial collapse that a 50/50 histogram
  cannot see.

### `acceptance`
- **Does**: Applies the thresholds recorded in `experiments/README.md` and emits
  named gates plus one aggregate decision.

### `acceptance_h1`
- **Does**: Adds density, part/checker balance, and zero exposed-assignment gates
  to the common structural criteria, including 0.80 spatial checker accuracy
  and a fully locked moving material state.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Experiment ledger | Stable JSON metric and gate names | Key/threshold changes |
| `experiment.py` | Pure read-only measurement | Side effects/signature |
