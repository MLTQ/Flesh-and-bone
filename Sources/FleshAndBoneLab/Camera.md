# `Camera.swift`

## Purpose

Provide the small orbit-camera math shared by interactive and future
headless/offscreen rendering without introducing a scene framework.

## Components

### `OrbitCamera`

- **Does:** Stores yaw, elevation, distance, and anatomical target; applies
  drag/scroll input; produces Metal right-handed view-projection plus billboard
  basis vectors.
- **Interacts with:** `FleshMetalView` input and `FleshRenderer`.
- **Depth direction:** Also supplies canonical splat sorting on camera changes.
- **Framing:** Defaults include the full 2 m character and splat tails at the
  app's wide and offscreen-test aspect ratios.

## Contracts

| Dependent | Expects | Breaking changes |
| --- | --- | --- |
| splat vertex shader | right/up vectors and Metal depth projection | Handedness/projection |
| app interaction | orbit uses pixel deltas; zoom is multiplicative | Input units |
