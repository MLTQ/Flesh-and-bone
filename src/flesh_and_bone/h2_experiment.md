# h2_experiment.py

## Purpose

Runs the full H2 development, wound, replacement-cell maturation, and motion
protocol for the primary arm and causal controls.

## Components

### `H2Config`
- **Does**: Records phase boundaries, feeding, rendering, field/control
  switches, mature-attachment deficit blend, and deterministic device/seed
  settings. H2 defaults the blend to zero so locked repair coordinates cannot
  be remapped by the developmental field. It records the stronger H2 local
  deficit contrast used to distinguish uncovered from occupied regional sites.

### `run_h2`
- **Does**: Grows generation 0, locks initial tissue, deletes the predeclared
  wound, re-feeds generation 1, enables local maturation, animates the repaired
  body, and writes phase metrics plus GIF/PNG/Markdown evidence.
- **Interacts with**: `H2BodyPlan`, `DeficitDynamics`, and `acceptance_h2`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `scripts/run_h2.py` | `run_h2(path, config)` returns a serializable report | Signature/config |
| Experiment ledger | Damage is measured before re-feeding; motion follows repair | Phase ordering |
| Lineage metrics | Initial cells are generation 0 and replacements generation 1 | Feed semantics |
