# `FleshSimulation.swift`

## Purpose

Own dynamic particle state and encode one exact 30 Hz H6C+H7C frame into a
Metal command buffer. Rig animation, four physics substeps, and profile-scaled
graph coupling stay in one auditable runtime boundary.

## Components

### `FleshSimulation`

- **Does:** Allocates twelve float4 state/intermediate buffers, compiles the
  three compute pipelines, exposes intent controls, and advances the model.
- **Interacts with:** immutable `RuntimeBody`/`RuntimeModel` and
  `FleshRenderer`.

### `encodeFrame`

- **Does:** Skins previous/current/next motion samples, then executes four
  neighbor-observation/integration pairs.
- **Rationale:** Four semi-implicit substeps at 30 Hz exactly match H7C.
- **Diagnostic override:** The optional substep count exists for parity
  localization; app and ordinary benchmarks always use four.

### `resetDynamics` / `stateSummary`

- **Does:** Restores cold zero residual/velocity and provides headless
  finite/RMS/max validation, including non-finite count and first cell index.

### `diagnosticState`

- **Does:** Reads one completed shared-buffer cell for parity diagnosis without
  adding a production GPU readback pass.

## Contracts

| Dependent | Expects | Breaking changes |
| --- | --- | --- |
| Metal shaders | `SimulationUniforms` field order and compute bindings | Layout/binding changes |
| renderer | current LBS and final ping-pong residual remain accessible | Buffer ownership |
| benchmark | one `encodeFrame` equals 1/30 simulated second | Timing/substep changes |
| control panel | speed, intensity, physics toggle, reset are live | Public controls |

## Notes

The constitutive neighbor coefficient scales by `(12.5 mm / pitch)²`, matching
the H5/H5D/H5U continuum-resolution convention. Physics-off still skins the
body but skips all neighbor and MLP work.
Simulation mode `2` retains only the H6C backbone as a live causal/debug
control; mode `1` is the full H6C+H7C model.
