# h5d_metrics.py

## Purpose

Applies H5D's cross-discretization gates to already validated H4 and H5 JSON
reports. It keeps density-scaling evidence distinct from H5's within-run gates.

## Components

### `evaluate_h5d`
- **Does**: Measures cell-count, teacher-continuity, and rollout comparisons and
  returns named gates plus an aggregate verdict.
- **Interacts with**: H4 volume reports and H5 teacher/run reports.
- **Rationale**: H5D-specific claims must be machine checked, not inferred from
  a passing H5 verdict alone.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `check_h5d.py` | JSON-serializable `metrics` and `gates` mappings | Key/schema changes |
| H5D ledger | Thresholds exactly match `experiments/H5D.md` | Threshold changes |
