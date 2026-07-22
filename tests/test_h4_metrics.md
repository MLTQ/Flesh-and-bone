# test_h4_metrics.py

## Purpose

Locks H4's multi-threshold influence-selection behavior independently of the
production asset.

## Contracts tested

| Contract | Expected | Regression caught |
|---|---|---|
| Influence selection | RMS, p99, and max must all pass; smallest passing K wins | Cherry-picking one aggregate metric |
