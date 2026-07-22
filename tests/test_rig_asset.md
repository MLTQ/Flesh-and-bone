# test_rig_asset.py

## Purpose

Provides CPU-fast contracts for H4 influence truncation and homogeneous linear
skinning independently of Blender or the generated rig bundle.

## Contracts tested

| Contract | Expected | Regression caught |
|---|---|---|
| Top-k weights | Retained mass is reported with matching dtype and selected weights renormalize to one | Silent mass loss/order error |
| Linear skinning | Bone transforms blend in canonical homogeneous coordinates | Matrix-axis/einsum mistakes |
