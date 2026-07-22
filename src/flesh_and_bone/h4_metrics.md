# h4_metrics.py

## Purpose

Centralizes H4's predeclared influence-selection and aggregate acceptance gates
so experiment orchestration cannot quietly weaken thresholds.

## Components

### `influence_pass` / `select_influence_count`
- **Does**: Applies RMS/p99/max transport limits and selects the smallest
  passing measured influence arm.

### `acceptance_h4`
- **Does**: Evaluates provenance, topology, canonical scale, full and selected
  skinning, surface completeness, and variable-thickness volume evidence.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| H4 experiment | Named gates exactly match `experiments/H4.md` | Threshold/key changes |
| Research ledger | Aggregate `pass` means every component passed | Gate removal |
