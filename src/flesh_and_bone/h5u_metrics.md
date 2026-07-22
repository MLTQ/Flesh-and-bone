# h5u_metrics.py

## Purpose

Applies H5U's ultra-density, H4 scope-exception, continuum, rollout, and frozen
appearance gates to canonical JSON reports.

## Components

### `evaluate_h5u`
- **Does**: Compares baseline H4/H5 with ultra-dense H4/H5U and returns named
  metrics, gates, and an aggregate verdict.
- **Rationale**: The historical H4 50k resource ceiling must remain failed while
  all structural H4 gates, H5 dynamics, and H5U render choices are checked
  independently.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `check_h5u.py` | JSON-serializable `metrics` and `gates` mappings | Key/schema changes |
| H5U ledger | Thresholds exactly match `experiments/H5U.md` | Threshold changes |
