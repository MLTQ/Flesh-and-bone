# test_h6m_metrics.py

## Purpose

Locks H6M's frozen-normalization diagnostic and strict inherited rollout gates.

## Contracts tested

| Contract | Expected | Regression caught |
|---|---|---|
| Feature sample | Requested examples are measured without changing buffers | Accidental renormalization |
| Acceptance | Good rollout still needs neighbor-causal degradation | Dropped ablation gate |
