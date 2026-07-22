# test_h4_volume.py

## Purpose

Provides CPU-fast geometry contracts for H4 bone-distance and voxel-component
measurements independently of the production asset.

## Contracts tested

| Contract | Expected | Regression caught |
|---|---|---|
| Segment distance | Projection clamps at bone endpoints | Infinite-line thickness error |
| Occupancy topology | Separate mass and enclosed empty pocket are distinguished | Misreported watertightness/connectivity |
| Barycentric UV | Projected points retain continuous per-triangle UV coordinates | Dense cells repeating nearest-vertex colors |
