# test_motion_variants.py

## Purpose

Locks the H6M temporal transformations before expensive frozen-checkpoint
evaluation.

## Contracts tested

| Contract | Expected | Regression caught |
|---|---|---|
| Reverse | Phase zero stays fixed; remaining frames reverse | Changed initialization pose |
| Periodic cubic | Integer upsample knots reproduce source exactly | Resampling phase drift |
| Frozen arms | Names/counts are `29,29,58,19,44` for LBS and bones | Ledger/code mismatch |
