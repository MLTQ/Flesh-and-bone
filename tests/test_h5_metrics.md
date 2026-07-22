# test_h5_metrics.py

## Purpose

Locks H5's aggregate acceptance behavior independently of the production asset
and documents that neither a worst-cell failure nor an ineffective neighbor
ablation may be hidden by good average error.

## Contracts tested

| Contract | Expected | Regression caught |
|---|---|---|
| Neighbor causality | Blind rollout must degrade position by 20% or edge strain by 25% | Model passing without using messages |
| Tail gate | Any maximum above 40 mm fails even when RMS/p99 are nominal | Rare surface outlier hidden by averages |
