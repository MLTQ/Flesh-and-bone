# `PopulationBrush.swift`

## Purpose

Convert a 2D viewer brush into directed anatomical niche selections without
coupling projection/depth policy to AppKit event handling or population state.

## Components

### `InteractionMode`

- **Does:** Distinguishes free orbit from directed NCA Source and NCA Vacuum
  painting.

### `PopulationBrush.selectTemplates`

- **Does:** Projects the current rendered template subset, applies a circular
  screen-space brush, then keeps only the frontmost pitch-scaled depth slab.
- **Source semantics:** Tests dormant capacity at each target niche and projects
  the animated target position, even though the new cell will spawn on bone.
- **Vacuum semantics:** Projects the highest active occupancy layer at each
  niche, including its current flesh residual.
- **Rationale:** A front-depth slab makes repeated strokes peel/add surface
  layers instead of modifying the entire body along an infinite view ray.

## Contracts

| Dependent | Expects | Breaking changes |
| --- | --- | --- |
| `FleshMetalView` | locations use ordinary lower-left AppKit view coordinates | Coordinate convention |
| `PopulationController` | templates expose source eligibility and active slots | Occupancy API |
| renderer | camera matrix and rendered template prefix match visible geometry | Projection/LOD |

## Notes

Selection scans the rendered template prefix on the CPU after the previous
Metal render completes. This intentionally trades a few milliseconds of brush
latency for race-free correspondence with the visible pose. It is an
interactive research tool, not the eventual GPU spatial-hash implementation.
