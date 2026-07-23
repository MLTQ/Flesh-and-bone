# run_h6c.py

## Purpose

Provides the reproducible CLI for H6C fitting, frozen motion evaluation,
ablation, metrics, and fast-arm visual evidence.

## Components

### `main`
- **Does**: Selects output/device while retaining the predeclared ten-cycle
  default, runs H6C, and prints its aggregate verdict.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| 4090 runner | `--device cuda` keeps fitting and rollouts on physical GPU | Device behavior |
| H6C ledger | Default horizon remains ten cycles | CLI/default changes |
