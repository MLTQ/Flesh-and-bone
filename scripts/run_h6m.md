# run_h6m.py

## Purpose

Provides the reproducible CLI for the H6M frozen-checkpoint motion
generalization experiment.

## Components

### `main`
- **Does**: Selects output/device/checkpoint directory, keeps the predeclared
  ten-cycle horizon and 1200 graph coupling by default, optionally suppresses
  only visual rendering, runs every motion arm, and prints the aggregate
  verdict.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| 4090 runner | `--device cuda` and H5U checkpoints require no retraining | CLI/default changes |
| H6M ledger | Default cycles are ten and coupling is 1200 | Physics/horizon changes |
| Headless diagnostics | `--skip-render` does not skip metrics or motion arms | Coupling rendering to evaluation |
