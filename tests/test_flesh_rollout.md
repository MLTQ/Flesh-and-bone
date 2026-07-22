# test_flesh_rollout.py

## Purpose

Locks H5's phase/substep/cell flattening and phase-held-out labeling convention.

## Contracts tested

| Contract | Expected | Regression caught |
|---|---|---|
| Teacher dataset | Example count is phase×substep×cell and labels remain phase-major | Train/holdout leakage or reshape error |
