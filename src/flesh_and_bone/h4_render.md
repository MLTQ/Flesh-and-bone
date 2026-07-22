# h4_render.py

## Purpose

Provides H4-specific texture sampling and fine colored-splat evidence rendering
without coupling the generic particle renderer to imported mesh assets.

## Components

### `load_base_color` / `sample_texture`
- **Does**: Reads the sole base-color texture directly from the immutable zip
  and bilinearly samples repeated Blender UV coordinates.
- **Rationale**: No extracted texture path or FBX material-node side effect is
  required at runtime.

### `render_colored_splats`
- **Does**: Projects canonical Y-up points, draws optional dynamic bones, and
  alpha-composites depth-sorted, individually scaled Gaussian cells.

### `error_colors`
- **Does**: Produces a fixed blue/white/red skinning-error diagnostic palette.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| H4 experiment | PIL images with fixed camera/color conventions | Projection/return changes |
| Source asset | Exactly one archive member ends in `_texture_0.png` | Texture naming |
| Volume renderer | Per-cell scales multiply a world-space base radius | Radius semantics |
