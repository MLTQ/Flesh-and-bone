# `FleshRenderer.swift`

## Purpose

Render the current LBS-plus-residual body as textured Gaussian splats with a
camera and visual density controls independent of physics resolution.

## Components

### `FleshRenderer`

- **Does:** Builds the premultiplied-alpha render pipeline and encodes one
  far-to-near instanced six-vertex billboard draw across both occupancy layers,
  masking dormant slots before fragment output.
- **Interacts with:** current buffers from `FleshSimulation`, immutable visual
  arrays from `RuntimeBody`, and `OrbitCamera`.

### `RenderUniforms`

- **Does:** Carries view projection, billboard basis, base/profile radius,
  user radius multiplier, opacity, template count, render-prefix count, and
  occupancy-layer count.

## Contracts

| Dependent | Expects | Breaking changes |
| --- | --- | --- |
| Metal shader | exact uniform and vertex-buffer indices | Layout/bindings |
| control panel | radius/count/opacity affect rendering only; population is physical | Physics coupling |
| CAMetalLayer | `.bgra8Unorm` drawable | Pixel format |

## Notes

The body updates canonical far-to-near order only after camera/count changes.
Instances are interleaved by sorted template, then occupancy layer, so paired
overcapacity cells remain adjacent in the blend order.
This removes unsorted transparency and depth-quad halos. Large deformation can
make canonical order approximate; production appearance may still prefer
weighted OIT.
