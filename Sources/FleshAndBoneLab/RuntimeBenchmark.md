# `RuntimeBenchmark.swift`

## Purpose

Measure native H6C+H7C compute scaling without a window, drawable throttling,
or Gaussian fill cost.

## Components

### `RuntimeBenchmark.run`

- **Does:** Loads each physical profile, warms Metal, advances 90 cold-start
  frames in one command buffer, and reports GPU ms/frame, 30 Hz headroom,
  static/dynamic memory, finite residual RMS/max, and non-finite cell count.
- **Diagnostic:** A failing row includes the first bad cell index for parity
  audits against the exported graph/material arrays and prints that cell's
  final residual, velocity, scalar features, and density vectors.
- **Interacts with:** the same `FleshSimulation` used by the app.

## Contracts

| Dependent | Expects | Breaking changes |
| --- | --- | --- |
| `main.swift` | `--benchmark [frames]` invokes this path | CLI changes |
| resource report | one frame contains four physics substeps | Integration timing |
| correctness check | final shared residual is readable after completion | Storage mode |

## Notes

Radius and render count are intentionally absent: they affect raster fill, not
the NCA rule. The interactive panel reports render timing separately.
Every row begins at the 100% reference population and includes the dormant
second-layer reserve plus its animated source-anchor skin sample, so its timing
matches the 200%-capable viewer. Kernels visit both layers even when the reserve
is inactive; the benchmark therefore reports the real capacity cost.
`FLESH_BACKBONE_ONLY=1` retains the same graph/integration path while zeroing
the learned density acceleration.
`FLESH_DEBUG_CELL=<index>` prints one final cell even when the row is finite.
`FLESH_DEBUG_LAST_SUBSTEPS=1...4` truncates only the final frame for locating a
shader-parity failure within its four integrations.
