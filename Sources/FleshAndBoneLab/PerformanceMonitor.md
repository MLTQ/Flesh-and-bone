# `PerformanceMonitor.swift`

## Purpose

Collect smoothed GPU compute/render timing and observed presentation rate
without stalling the command queue.

## Components

### `PerformanceMonitor`

- **Does:** Consumes command-buffer GPU timestamps from completion handlers,
  tracks a one-second presentation window, and returns locked snapshots.
- **Rationale:** Compute and rendering use separate serial command buffers so
  count/radius fill cost is not confused with NCA physics cost.

### `PerformanceSnapshot`

- **Does:** Reports milliseconds, FPS, and 30 Hz physics headroom.

## Contracts

| Dependent | Expects | Breaking changes |
| --- | --- | --- |
| `FleshMetalView` | completion callbacks never block | Synchronous work |
| `ControlPanel` | snapshot fields remain display-safe from main thread | Field semantics |

## Notes

GPU timestamps can be unavailable on the first submitted buffer; zero is shown
until a valid sample arrives.
