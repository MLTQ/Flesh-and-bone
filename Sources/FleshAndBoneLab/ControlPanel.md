# `ControlPanel.swift`

## Purpose

Expose the experiment's independent physical, visual, intent, and runtime
controls without hiding which sliders alter mechanics.

## Components

### `ControlPanel`

- **Does:** Builds the programmatic AppKit panel, binds profile/count/radius/
  opacity/speed/intensity/physics/density/reset/pause controls, and refreshes
  metrics.
- **Implementation:** Sliders are created without targets, then bound after
  `self` is initialized.
- **Interacts with:** `FleshMetalView` and `PerformanceMonitor`.

### physical-profile control

- **Does:** Rebuilds a cold simulation for 13k, 35k, or 92k cells.
- **Rationale:** A new count means a new pitch and graph, not an instance limit.

### rendered-cell/radius controls

- **Does:** Change only instance count and Gaussian fill footprint.
- **Rationale:** Keeps rendering performance separate from physics claims.
- **Ordering:** Render-count changes also re-sort that uniform sample prefix.

## Contracts

| Dependent | Expects | Breaking changes |
| --- | --- | --- |
| app user | labels distinguish physical and visual controls | Semantic relabeling |
| `FleshMetalView` | profile/settings/reset APIs are main-thread safe | Public API |
| performance report | static/dynamic MB and GPU timings retain units | Unit changes |

## Notes

The current broadcast intent has two dimensions—speed and intensity. This is a
deliberately small placeholder for an eventual LLM→intent→RL skeleton policy.
The density checkbox is a matched H6C-backbone causal control, not a visual
effect.
