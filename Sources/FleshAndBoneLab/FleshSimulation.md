# `FleshSimulation.swift`

## Purpose

Own dynamic particle state and encode one exact 30 Hz H6C+H7C frame into a
Metal command buffer. Rig animation, four physics substeps, and profile-scaled
graph coupling stay in one auditable runtime boundary.

## Components

### `FleshSimulation`

- **Does:** Allocates thirteen float4 state/intermediate buffers plus an active
  scalar for two slots per anatomical niche, compiles four compute pipelines,
  exposes intent/population controls, and advances the model.
- **Interacts with:** immutable `RuntimeBody`/`RuntimeModel` and
  `FleshRenderer`.

### `encodeFrame`

- **Does:** Skins previous/current/next motion samples, then executes four
  neighbor-observation/integration pairs. The same skin pass also animates
  dominant-bone source anchors.
- **Rationale:** Four semi-implicit substeps at 30 Hz exactly match H7C.
- **Diagnostic override:** The optional substep count exists for parity
  localization; app and ordinary benchmarks always use four.

### `resetDynamics` / `stateSummary`

- **Does:** Restores cold zero residual/velocity and provides headless
  finite/RMS/max validation, including non-finite count and first cell index.
  State summaries exclude dormant cells.

### population planning / `encodePopulationChange`

- **Does:** Plans arbitrary directed slot changes and queue-encodes their state
  mutation. Source cells start at the moving bone anchor with zero velocity;
  vacuumed cells clear all ping-pong state.
- **Interacts with:** `PopulationController`, `FleshMetalView`, and the Metal
  population kernel.

### `diagnosticState`

- **Does:** Reads one completed shared-buffer cell for parity diagnosis without
  adding a production GPU readback pass.

## Contracts

| Dependent | Expects | Breaking changes |
| --- | --- | --- |
| Metal shaders | `SimulationUniforms` field order and compute bindings | Layout/binding changes |
| renderer | current LBS and final ping-pong residual remain accessible | Buffer ownership |
| benchmark | one `encodeFrame` equals 1/30 simulated second | Timing/substep changes |
| control panel | speed, intensity, physics, population, and reset controls are live | Public controls |

## Notes

The constitutive neighbor coefficient scales by `(12.5 mm / pitch)²`, matching
the H5/H5D/H5U continuum-resolution convention. Physics-off still skins the
body but skips all neighbor and MLP work.
Simulation mode `2` retains only the H6C backbone as a live causal/debug
control; mode `1` is the full H6C+H7C model.
Layer zero starts as the complete 100% body and layer one as dormant reserve,
for a 200% ceiling. Both layers run the same fixed graph independently; paired
active layers also exchange one explicit density observation at their shared
niche. Population changes do not allocate new topology or infer new fate.
