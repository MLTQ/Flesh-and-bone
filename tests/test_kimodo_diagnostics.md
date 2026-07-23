# `test_kimodo_diagnostics.py`

## Purpose

Verify gross-anatomy gates and diagnostic rendering helpers with a small
synthetic humanoid hierarchy, independent of Kimodo, Torch, and Pillow.

## Components

### Neutral and malformed skeleton tests

- **Does:** Confirms a conventional stance passes while a steep pelvis/raised
  socket configuration fails.

### Contact and helper tests

- **Does:** Confirms twelve-centimeter planted-foot drift still fails after the
  intentional three-frame release blend is excluded, helper segments can be
  omitted, and bilateral limbs receive distinct colors.

## Contracts

| Dependent | Expects | Breaking changes |
| --- | --- | --- |
| `kimodo_diagnostics.py` | verdict/status/unit schema and named Meshy bones | Threshold or return-schema changes |
| test fixtures | endpoints use `[frames,bones,2,3]` | Shape changes |
