# `MetalShaders.swift`

## Purpose

Define the complete native GPU contract: rig skinning, fixed-neighbor
observation, exact H6C+H7C integration, and Gaussian-splat rendering.

## Components

### `skin_motion`

- **Does:** Evaluates six-weight LBS at previous/current/next fractional phases
  plus each cell's bone source anchor and applies the live motion-intensity
  intent.
- **Layering:** Maps every dynamic slot to its immutable anatomical template,
  so reserve cells receive the same rig target and source anchor.
- **Rationale:** Three samples provide nonperiodic-equivalent central
  acceleration without uploading per-cell animation frames.

### `observe_neighbors`

- **Does:** Computes graph Laplacians, axial compression/stretch statistics, and
  explicit equivariant density vectors over active neighbors only.
- **Overcapacity:** When both slots at one niche are active, adds an
  antisymmetric pitch-scaled separation observation. This gives the existing
  bounded density rule a real local compression signal instead of rendering
  two noninteracting duplicates.
- **Isolation:** Paired occupancy does not enter the H6C graph-Laplacian means;
  the live density checkbox is therefore a matched causal control for this
  added observation.

### `integrate_flesh`

- **Does:** Applies the five-term constitutive backbone and the exact
  5→32→32→2 bounded H7C residual, then advances semi-implicit Euler.
- **Rationale:** The learned head chooses only pressure/cohesion magnitudes;
  directions and the 12 m/s² cap remain structural.

### `apply_population_change`

- **Does:** Activates or deactivates arbitrary brush-selected slots; source cells
  begin at their animated bone anchors with zero velocity, while vacuumed cells
  clear both ping-pong states.
- **Rationale:** Queue-ordered GPU mutation avoids racing live render/physics
  buffers from AppKit callbacks.

### `splat_vertex` / `splat_fragment`

- **Does:** Draws camera-facing Gaussian quads using deterministic render-order
  prefixes, active-population masking, and premultiplied alpha.

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
Dormant cells do not contribute neighbor messages, integrate, or render, though
the fixed-capacity kernels still visit their slots. Their exported niches and
graph edges remain allocated. The second occupancy layer makes local 200%
stress possible, but this remains fixed topology rather than cell division,
fate inference, or spatial-neighbor insertion.
The smooth density cap branches explicitly at zero norm. This is algebraically
identical to the PyTorch epsilon form for zero vectors and avoids a GPU
zero-times-ratio edge case.
Its `tanh` argument is clamped at 10, where float32 has already rounded the
result to one. This preserves PyTorch values while avoiding Apple fast-math
overflow for severely compressed coarse cells.
