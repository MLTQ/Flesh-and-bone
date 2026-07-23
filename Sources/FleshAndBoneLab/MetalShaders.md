# `MetalShaders.swift`

## Purpose

Define the complete native GPU contract: rig skinning, fixed-neighbor
observation, exact H6C+H7C integration, and Gaussian-splat rendering.

## Components

### `skin_motion`

- **Does:** Evaluates six-weight LBS at previous/current/next fractional phases
  and applies the live motion-intensity intent.
- **Rationale:** Three samples provide nonperiodic-equivalent central
  acceleration without uploading per-cell animation frames.

### `observe_neighbors`

- **Does:** Computes graph Laplacians, axial compression/stretch statistics, and
  explicit equivariant density vectors over at most six neighbors.

### `integrate_flesh`

- **Does:** Applies the five-term constitutive backbone and the exact
  5→32→32→2 bounded H7C residual, then advances semi-implicit Euler.
- **Rationale:** The learned head chooses only pressure/cohesion magnitudes;
  directions and the 12 m/s² cap remain structural.

### `splat_vertex` / `splat_fragment`

- **Does:** Draws camera-facing Gaussian quads using deterministic render-order
  prefixes and premultiplied alpha.

## Contracts

| Dependent | Expects | Breaking changes |
| --- | --- | --- |
| `FleshSimulation.swift` | compute buffer indices and uniform layout | Binding changes |
| `FleshRenderer.swift` | render bindings, premultiplied output | Vertex/fragment contract |
| Python H7C | thresholds, features, SiLU MLP offsets, smooth cap | Math/order changes |
| runtime exporter | Metal-column-major skin matrices | Matrix orientation |

## Notes

Physics resolution changes the graph and pitch-scaled neighbor coefficient.
Render count only shortens the render-order prefix; it never changes physics.
Function-local offsets and corner tables use ordinary `const`; Metal's
`constant` qualifier is reserved for address-space declarations.
Simulation mode `2` zeros the learned density acceleration after evaluating the
same observations, providing a matched H6C-backbone control.
The smooth density cap branches explicitly at zero norm. This is algebraically
identical to the PyTorch epsilon form for zero vectors and avoids a GPU
zero-times-ratio edge case.
Its `tanh` argument is clamped at 10, where float32 has already rounded the
result to one. This preserves PyTorch values while avoiding Apple fast-math
overflow for severely compressed coarse cells.
