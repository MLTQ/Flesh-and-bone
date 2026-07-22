# deficit_dynamics.py

## Purpose

Implements H1 continuous tissue-deficit recruitment without exposing any
particle to a target-site index. It also provides the pressure-off and
nearest-bone-only causal controls.

## Components

### `DeficitDynamicsConfig`
- **Does**: Names occupancy, recruitment, deficit, attachment, density,
  integration, local-maturation, and learned-residual scales.

### `DeficitDynamics.step`
- **Does**: Rasterizes particle occupancy onto body-plan samples, computes
  unfilled demand, lets every cell attend continuously to that field, applies
  density pressure/anti-overlap, differentiates near the body, and switches
  committed cells to persistent bone attachment during motion.
- **Rationale**: Reference samples define the developmental field and metrics,
  but cells never reserve or receive a discrete sample identity.
- **Material phase**: While the scaffold is in its assembly pose, contacted but
  unlocked cells can refresh checker phenotype from the nearest local sample of
  the material morphogen. The experiment locks this state before motion; no
  target identity is retained.
- **Local maturation**: H2 may require consecutive low-speed, near-target-density
  steps before replacement material locks. Locked cells use their attachment
  target even while a wound is repaired at rest.

### Recruitment controls
- **Does**: `deficit` follows live tissue demand; `nearest_bone` ignores body
  thickness/demand and collapses toward skeleton segments. Pressure can be
  disabled independently. H2 can also deny the nearest-bone arm spatial target
  density, giving it only one uniform pressure set point.
- **Hierarchical transport**: `hierarchical_deficit` allocates one hard
  anatomical-region guide from unfilled regional capacity and distance, then
  uses a broader continuous attention field only inside that region. The coarse
  guide is stored hysteretically; no target-site identity or final position is
  supplied. Removing cells frees regional capacity for replacement lineage.
  Attention is broad only while a cell is in transit and contracts to the H1
  local radius on arrival, preventing each region from collapsing to its
  centroid.
- **Learned selector hook**: A caller may replace only the hard regional choice
  with a selector callback receiving shortage, target capacity, and cell-region
  distance. H3 uses this hook for the frozen fate MLP while preserving identical
  within-region physics. H1/H2 retain the deterministic default.
- **Locality invariant**: Deficit attention is deliberately narrower than an H
  branch separation. A wide normalized attention field averages symmetric
  unfilled branches into an invalid target in the central negative space.
- **Motion invariant**: Once differentiated, cells primarily follow their
  persistent soft-bone coordinates. Only a small live-deficit correction is
  retained during articulation so the tissue field cannot remap body regions.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `h1_experiment.py` | Named diagnostics, plastic-material switch, and deterministic mutation step | Signature/metric names |
| `ParticleNCARule` | Same 12 feature channels as H0 | Feature order/count |
| Experiment ledger | Cells have no assigned sites in every H1/control arm | Introducing discrete assignment |
| H3 fate model | Optional selector controls only new coarse guide commitments | Callback inputs/timing |
