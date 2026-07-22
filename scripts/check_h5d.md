# check_h5d.py

## Purpose

Provides the reproducible post-run comparison for H5D, combining baseline and
dense H4/H5 reports into one machine-readable density verdict.

## Components

### `main`
- **Does**: Loads four metrics files, evaluates H5D gates, writes the comparison
  JSON, and prints a compact verdict.
- **Interacts with**: `evaluate_h5d` in `h5d_metrics.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| H5D reproduction | Defaults match canonical run directories | Default paths |
| Research audit | Writes `density_acceptance.json` with named gates | Output schema |
