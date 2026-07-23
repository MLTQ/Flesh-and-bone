# `RuntimeRenderTest.swift`

## Purpose

Validate and benchmark the exact app render pipeline without screen-capture,
Accessibility, window-server timing, or presentation throttling.

## Components

### `render`

- **Does:** Advances the chosen profile for 30 physics frames, renders one
  1100×760 BGRA8 frame with production canonical depth sorting, reads it back,
  and writes a PNG.
- **Interacts with:** the same `FleshSimulation` and `FleshRenderer` as the app.
- **View control:** Accepts the same camera presets and opacity as the app so
  front/back visibility reports exercise the reported configuration.
- **Implementation:** Readback swizzles BGRA to RGBA before copying into
  `NSBitmapImageRep`; `swapAt` preserves Swift's exclusive-access rules.

### `benchmark`

- **Does:** Separately measures render GPU time across all profiles, three
  radius multipliers, and 25/50/100% render-count arms for the 92k body.
- **Rationale:** Radius changes fragment fill while physical profile changes
  compute and geometry; both need explicit cost curves.
- **Warmup:** Submits one full-count render per profile before timing so shader
  compilation and initial GPU frequency ramp do not contaminate the first arm.

## Contracts

| Dependent | Expects | Breaking changes |
| --- | --- | --- |
| `main.swift` | `--render-test [png] [preset] [opacity]` and benchmark flags | CLI changes |
| visual QA | PNG uses the production renderer after real physics | Alternate path |
| resource notes | benchmark target stays 1100×760 | Target-size changes |
