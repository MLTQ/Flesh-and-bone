# `runtime_export.py`

## Purpose

Export compact binary assets for the native Metal runner. The exporter keeps
research NPZ/PyTorch formats out of Swift while preserving exact body, graph,
material, rig-animation, and H7C weight semantics.

## Components

### `export_runtime_body`

- **Does:** Writes one `FNB1` body containing aligned rest points, six skin
  influences, dominant-bone source anchors, material fields, fixed six-neighbor
  topology, RGBA8 texture, deterministic render order, and the rig's unique
  walk matrices.
- **Rationale:** Physics resolution and render-count sampling remain separate.
  A deterministic random render order gives approximately uniform visual
  coverage at every prefix length.

### `_bone_source_anchors`

- **Does:** Projects each cell's target point onto its dominant weighted bone
  segment.
- **Rationale:** Source activation begins on the moving skeleton without
  inventing a second anatomical identity or exposing a world-axis shortcut.

### `export_runtime_model`

- **Does:** Writes one `FNM1` model containing the five H6C coefficients,
  H7C coefficient bounds, and the 5→32→32→2 SiLU MLP.
- **Rationale:** The learned payload is independent of body resolution.

### `export_runtime_bundle`

- **Does:** Exports every declared body plus one shared model and JSON manifest.
- **Interacts with:** `scripts/export_flesh_runtime.py` and the Swift
  `RuntimeAsset` loader.

## Contracts

| Dependent | Expects | Breaking changes |
| --- | --- | --- |
| `RuntimeAsset.swift` | `FNB1` v2/`FNM1`, little-endian headers, exact array order | Header or array layout |
| Metal kernels | points/material use float4; neighbors have eight int32 lanes | Alignment or lane count |
| renderer | colors are RGBA8 and render order is a complete permutation | Color/order encoding |
| physics | matrices are transposed for Metal column-major multiplication | Matrix orientation |

## Notes

Missing neighbors are `-1`. The final two influence lanes are zero-padded.
Body assets are generated build products; canonical NPZ assets remain source.
The loader retains a v1 fallback where the source anchor equals the target, but
only v2 provides bone-origin feeding.
