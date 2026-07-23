# `FleshMetalView.swift`

## Purpose

Join the native simulation, renderer, CAMetalLayer, 30 Hz scheduler, profile
hot-swap, orbit input, and directed population painting in one AppKit view.

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

- **Does:** Orbits/zooms in Orbit mode or projects a visible surface brush and
  submits Source/Vacuum changes in the corresponding paint mode.
- **Ordering:** The brush waits for the last render command before reading
  shared pose buffers. Population edits are then submitted on the same serial
  queue as simulation and rendering.

### `setCameraPreset`

- **Does:** Snaps to a named anatomical view and immediately rebuilds the
  far-to-near splat order.
- **Interacts with:** the control panel's Front/Left/Back/Right buttons.

### directed population painting

- **Does:** Selects only the frontmost pitch-scaled depth slab under a circular
  screen-space brush. One continuous drag changes each niche at most once.
- **Rationale:** A directed tool can test local wounds and local overcapacity;
  a global percentage button cannot. Queue ordering prevents AppKit callbacks
  from racing live Metal buffers.

### `reset`

- **Does:** Recreates the dynamic simulation at the 100% baseline while
  preserving profile, motion settings, and causal-control settings.

## Contracts

| Dependent | Expects | Breaking changes |
| --- | --- | --- |
| `ControlPanel` | public settings, profile list/index, camera/population controls, simulation metrics | Property API |
| CAMetalLayer | BGRA8, 30 Hz drawable submissions | Pixel/timing changes |
| performance UI | compute and render are distinct command buffers | Queue structure |

## Notes

The timer runs in common run-loop mode so dragging sliders does not pause the
organism. A production desktop pet may use CVDisplayLink and decouple 60 Hz
rendering from 30 Hz physics.
Pause stops physics advancement but still renders, so population edits remain
visible on a frozen pose.
