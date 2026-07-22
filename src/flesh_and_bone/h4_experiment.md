# h4_experiment.py

## Purpose

Runs H4's Blender-independent production-rig round-trip, influence ablations,
fine textured surface transport, and watertight variable-thickness volume
construction/animation.

## Components

### `H4Config`
- **Does**: Records immutable archive/derived-rig paths, voxel pitch, render
  resolution/radii, and GIF timing. The frozen H4 volume radius is `0.30 ×
  pitch` before per-material scale, chosen after the first `0.65 × pitch`
  evidence render obscured individual cells despite correct geometry.

### `run_h4`
- **Does**: Loads canonical rig data; measures full/top-3/top-4/top-6 skinning;
  selects the smallest passing arm; builds and skins the volume; evaluates all
  gates; and writes metrics, volume, contact sheets, GIFs, and error image.
- **Interacts with**: Rig loader, H4 volume, H4 metrics, texture renderer, and
  generic artifact writers.

### `_skin_in_frames`
- **Does**: Bounds intermediate memory by applying LBS one frame at a time,
  which matters for the 13k-cell volume.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `scripts/run_h4.py` | `run_h4(path, config)` returns a serializable report | Signature/config |
| H4 ledger | Thresholds are applied before visuals are written | Gate semantics |
| Future trainers | Run-local `volume.npz` is versioned and carries six-weight state | Volume format/order |

## Notes

Texture colors are inspection evidence. Averaged per-vertex UVs can blur seams;
exact production material transfer remains per-corner or per-surface-sample.
