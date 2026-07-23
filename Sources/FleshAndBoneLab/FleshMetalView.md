# `FleshMetalView.swift`

## Purpose

Join the native simulation, renderer, CAMetalLayer, 30 Hz scheduler, profile
hot-swap, and orbit input in one AppKit view.

## Components

### `FleshMetalView`

- **Does:** Resolves the shared model/profiles, owns one serial Metal queue, and
  submits separate compute and render command buffers every 1/30 second.
- **Interacts with:** control panel settings and `PerformanceMonitor`.
- **Ordering:** Re-sorts the selected canonical splats only after orbit,
  profile, or render-count changes.
- **Default:** Opens the highest-density profile so the first view exposes the
  best available face and clothing detail; lower-density profiles remain a
  one-click physical comparison.

### `loadProfile`

- **Does:** Replaces the physical graph/body and cold-resets dynamics while
  retaining camera and visual settings.
- **Rationale:** Physics resolution is a real state reset, not a render LOD.

### mouse/scroll handling

- **Does:** Orbits and zooms without changing simulation state.

### `setCameraPreset`

- **Does:** Snaps to a named anatomical view and immediately rebuilds the
  far-to-near splat order.
- **Interacts with:** the control panel's Front/Left/Back/Right buttons.

## Contracts

| Dependent | Expects | Breaking changes |
| --- | --- | --- |
| `ControlPanel` | public settings, profile list/index, camera presets, simulation metrics | Property API |
| CAMetalLayer | BGRA8, 30 Hz drawable submissions | Pixel/timing changes |
| performance UI | compute and render are distinct command buffers | Queue structure |

## Notes

The timer runs in common run-loop mode so dragging sliders does not pause the
organism. A production desktop pet may use CVDisplayLink and decouple 60 Hz
rendering from 30 Hz physics.
