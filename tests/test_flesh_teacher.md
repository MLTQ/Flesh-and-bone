# test_flesh_teacher.py

## Purpose

Locks H5's voxel-neighbor graph direction, degree, connectivity, and local mean
difference convention without requiring the production volume.

## Contracts tested

| Contract | Expected | Regression caught |
|---|---|---|
| Three-cell line | Degrees are 1/2/1 and endpoint messages point inward | Reversed edges or unnormalized accumulation |
