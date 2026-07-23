# run_h6k.py

## Purpose

Provides the reproducible CLI for H6K bridge validation, frozen MLP/H6C
rollouts, controls, metrics, and visual evidence.

## Components

### `main`
- **Does**: Selects device/output plus an optional prepared motion artifact,
  runs the fixed protocol, and prints its three explicit verdicts.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| 4090 runner | `--device cuda` runs all dense teachers/rollouts on the 4090 | Device behavior |
| Research ledger | No CLI switch changes cycles, seeds, or gates | Hidden protocol override |
