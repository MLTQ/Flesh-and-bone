# extract_h4_asset.py

## Purpose

Runs inside Blender to convert the supplied static/animated FBXs into one
portable, canonical NumPy rig asset. It isolates FBX SDK conventions from the
runtime and records enough Blender-evaluated ground truth to audit skinning.

## Components

### Canonical transform
- **Does**: Maps Blender world `(x, y, z)` to right-handed experiment space
  `(x, z, -y)` in meters, making the humanoid's vertical axis `y`.

### `_base_mesh`
- **Does**: Extracts world-space bind vertices, exact triangles, per-corner UVs,
  dense ordered skin weights, bone parents, rest endpoints, and bind matrices.
- **Rationale**: Dense 24-bone extraction is the lossless oracle; top-k is a
  later measured deployment choice.

### `_animation`
- **Does**: Exports every action frame's Blender-evaluated vertices, canonical
  skin matrices, and posed bone endpoints.
- **Rationale**: Evaluated vertices make matrix/order/axis mistakes directly
  measurable without Blender at runtime.

### `main`
- **Does**: Hashes and extracts the immutable zip, validates static/animated
  topology and bone order, writes a compressed `.npz`, and prints provenance.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| H4 rig loader | `flesh-and-bone-h4-rig-v1` arrays and canonical coordinates | Key/shape/axis changes |
| H4 ledger | Archive hash/member names and extraction counts are recorded | Metadata removal |
| Blender runner | Arguments follow `-- --archive ... --output ...` | CLI syntax |

## Notes

This script intentionally requires Blender's `bpy`. It is not imported by the
ordinary Python package or CPU test suite.
