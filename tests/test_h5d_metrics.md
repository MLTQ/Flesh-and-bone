# test_h5d_metrics.py

## Purpose

Locks H5D's density and continuum-comparison gates without requiring generated
assets or a GPU.

## Contracts tested

| Contract | Expected | Regression caught |
|---|---|---|
| Valid scaling | Dense H4/H5 reports satisfying every threshold pass | Broken aggregate logic |
| Teacher continuity | A more-dense but physically mismatched teacher fails | Density mistaken for equivalent mechanics |
