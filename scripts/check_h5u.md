# check_h5u.py

## Purpose

Provides the reproducible post-run H5U comparison, including the intentional
H4 resource-scope failure and the frozen visual configuration.

## Components

### `main`
- **Does**: Loads canonical baseline/ultra reports, evaluates H5U, writes
  `ultra_acceptance.json`, and prints a compact verdict.
- **Interacts with**: `evaluate_h5u` in `h5u_metrics.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| H5U reproduction | Defaults match canonical run directories | Default paths |
| Research audit | Named scope, density, physics, and render gates | Output schema |
