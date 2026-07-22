# h4_volume.py

## Purpose

Builds H4's watertight, fine variable-thickness developmental target from the
imperfect source surface and transfers rig/material state without pretending
that transfer is learned differentiation.

## Components

### `H4Volume`
- **Does**: Stores occupied voxel centers, top-k skin weights, dominant bone,
  nearest source vertex, UV, bone distance, and material splat scale.

### `point_segment_distance`
- **Does**: Measures exact clamped distance from every cell to the nearest rest
  bone segment for extra-skeletal-volume evidence.

### `occupancy_components`
- **Does**: Uses 6-connectivity to count occupied components and empty
  components not connected to the grid boundary.

### `build_h4_volume`
- **Does**: Surface-voxelizes and flood-fills at configured pitch, transfers
  nearest-vertex UV/top-k skin state, and records topology/thickness metrics.
  The configured base splat-radius scale is included in the maximum world-radius
  evidence rather than duplicated in the renderer.
- **Rationale**: Voxel filling tolerates the source's small boundary and
  non-manifold defects; ordinary signed-distance containment would not.

### `save_h4_volume` / `load_h4_volume`
- **Does**: Round-trip a versioned compressed volume without pickle.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| H4 experiment | Metadata contains cell/connectivity/thickness/splat gates | Key semantics |
| Animated volume | Weights use the rig asset's exact bone order | Bone reordering |
| Future developmental trainer | Points are canonical rest targets, not preanimated coordinates | Point semantics |

## Notes

Nearest-surface weight and UV transfer is an initialization oracle. Future
learned cells must infer analogous state from developmental signals.
