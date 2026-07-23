# `RuntimeAsset.swift`

## Purpose

Load the compact `FNB1` body profiles and shared `FNM1` model into Metal
buffers without teaching Swift about NPZ, PyTorch, textures, or graph building.

## Components

### `RuntimeBody`

- **Does:** Validates one body header and maps every aligned array into a shared
  Metal buffer.
- **Interacts with:** `FleshSimulation` and `FleshRenderer`.
- **Population state:** v2 bodies carry one dominant-bone source anchor per
  cell. Occupancy layers are runtime state and do not duplicate the body asset.

### `RuntimeBody.sortRenderOrder`

- **Does:** Takes the deterministic uniform-sample prefix and sorts it
  canonical far-to-near for the current camera.
- **Rationale:** Soft premultiplied splats need ordering; sorting only after
  camera/count changes avoids a per-frame GPU sort.

### `RuntimeModel`

- **Does:** Validates the fixed 5→32→32→2 H7C architecture and loads its flat
  backbone/residual payload.
- **Interacts with:** Metal integration kernel offsets.

### `RuntimeAssets`

- **Does:** Resolves assets from `FLESH_ASSETS_DIR`, repository
  `runtime/Assets`, or app-bundle Resources and discovers profiles by count.

## Contracts

| Dependent | Expects | Breaking changes |
| --- | --- | --- |
| Python exporter | little-endian `FNB1` v1/v2 and `FNM1` with exact array order | Format changes |
| Metal shaders | float4 points/material, 8 influence/neighbor lanes | Buffer alignment |
| control panel | profile paths sort from lowest to highest cell count | Naming convention |

## Notes

All buffers use shared storage for portability across Apple silicon. The
runtime owns immutable body/model buffers; simulations own only dynamic state.
Version 1 bodies remain loadable but alias source anchors to final rest points,
so they cannot visibly demonstrate skeleton-origin feeding. Current exports are
version 2.
