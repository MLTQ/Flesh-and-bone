# test_h5u_metrics.py

## Purpose

Locks H5U's intentional H4 resource exception and its ultra-density, continuous
UV, continuum, rollout, and render gates without GPU assets.

## Contracts tested

| Contract | Expected | Regression caught |
|---|---|---|
| H4 scope | Only `volume_cell_count` and aggregate `pass` may be false | Structural failure misreported as intentional scaling |
| Continuous UV | Nearest-vertex transfer fails H5U | Cell count claimed as texture detail without new samples |
