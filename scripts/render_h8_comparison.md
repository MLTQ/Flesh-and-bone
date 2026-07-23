# `render_h8_comparison.py`

## Purpose

Render the exact visual comparison motivated by H8: rig-only LBS beside the
same retargeted motion with frozen H7C mechanics. Labels state whether physics
is off or on so the control cannot be confused with the research arm.

## Components

### `_render_state`

- **Does:** Loads one bounded seed-7 render state, follows Hips in X/Z, renders
  texture-identical LBS/hybrid panels, and writes a GIF plus contact sheet.
- **Interacts with:** H4 texture/splat rendering, imported rig parents, and H8
  render states from `h8_experiment.py`.

### `main`

- **Does:** Discovers all qualification/final render states and renders them with
  one shared body, texture, radius, opacity, and panel size.

## Contracts

| Dependent | Expects | Breaking changes |
| --- | --- | --- |
| H8 reviewer | left is zero-residual LBS, right is actual unscaled hybrid | Panel semantics or residual scaling |
| render state | `flesh-and-bone-h8-render-v1` tensors | Field names/shapes |
| contact sheet | up to eight evenly spaced paired frames | Sampling/layout changes |

## Notes

Rendering uses every fourth cell and at most 60 frames saved by the experiment.
This affects visual cost only; metrics use every cell and every stream frame.
