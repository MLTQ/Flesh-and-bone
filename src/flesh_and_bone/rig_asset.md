# rig_asset.py

## Purpose

Loads the Blender-independent H4 rig bundle and implements auditable linear
skinning. Full and top-k influence arms share the same code path so deployment
compression cannot hide an import error.

## Components

### `RigAsset` / `load_rig_asset`
- **Does**: Holds canonical mesh, topology, UVs, ordered bones, dense weights,
  bind metadata/rest endpoints, per-frame skin matrices/endpoints, and Blender
  vertex oracle.
- **Rationale**: Loading rejects unknown format magic and avoids pickle/object
  arrays so generated assets are deterministic data rather than executable
  state.

### `normalized_topk_weights`
- **Does**: Keeps the largest requested influences, reports retained mass, and
  renormalizes exactly. `None` retains the full dense oracle.

### `linear_skin`
- **Does**: Applies canonical homogeneous skin matrices and blends transformed
  points with ordered weights for every animation frame.

### `evaluate_skinning`
- **Does**: Reports retained weight, global/per-frame RMS, p99, max, and finite
  state against Blender-evaluated vertices.

### `averaged_vertex_uv`
- **Does**: Produces inspection-only per-vertex UVs by averaging triangle-corner
  values. Exact seam-aware material sampling remains a per-corner operation.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| H4 experiment | Stable metric keys and `(frames, vertices, 3)` skin output | Shape/key changes |
| Volume transfer | Ordered bone names/weights remain canonical | Bone reordering |
| Visual renderer | Averaged UVs are explicitly approximate at seams | UV semantics |
