# h1_experiment.py

## Purpose

Runs H1 and its causal controls with no per-particle destination identities,
smaller visible splats, continuous tissue-deficit recruitment, two-stage
plastic/locked material identity, and the common evidence pipeline.

## Components

### `H1Config`
- **Does**: Records schedule, geometry, smaller splat scale, device, pressure
  switch, recruitment arm, and material-plasticity switch in every metrics file.

### `run_h1`
- **Does**: Feeds unassigned cells, advances deficit dynamics, locks plastic
  phenotype/attachments at motion onset, measures H1-specific gates, and writes
  JSON/GIF/PNG/Markdown artifacts.
- **Interacts with**: `DeficitDynamics`, `ParticleSystem`, and `acceptance_h1`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `scripts/run_h1.py` | `run_h1(path, config)` returns one report | Signature/config |
| Experiment ledger | Main/legacy-texture/pressure-off/nearest-bone arms share schedule/seed | Output schema |
