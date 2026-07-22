# experiment.py

## Purpose

Orchestrates the deterministic H0 feed, assembly, differentiation, skeletal
motion, measurement, rendering, and evidence-writing schedule.

## Components

### `ExperimentConfig`
- **Does**: Captures every run-defining seed, schedule, geometry, rendering, and
  device option written to the metrics record.

### `run_h0`
- **Does**: Builds the body plan/reservoir, activates a zero-output NCA residual,
  feeds cells, runs mechanics, records assembly and moving metrics, applies
  acceptance gates, scales splat rendering from body-plan spacing, and writes
  JSON/GIF/PNG/Markdown evidence.
- **Rationale**: The neural module is present but exactly inert, making H0 an
  honest mechanical control for H1. Contact-sheet milestones include half-fed,
  fully-fed, settled, early-motion, and final states so migration is visible.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `scripts/run_h_baseline.py` | `run_h0(path, config)` returns report | Signature/config fields |
| Experiment ledger | Stable artifact names and resolved JSON config | Output schema/names |
