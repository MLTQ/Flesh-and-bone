# morphology.py

## Purpose

Builds the explicit H0 body-plan control: regularly spaced niches inside the
union of five bone capsules, persistent checker identity, soft skeletal
attachment, target neighbor density, and two negative-space probes.

## Components

### `BodyPlan`
- **Does**: Holds immutable niche positions, bone-local embeddings, identities,
  density targets, and experiment geometry.

### `checker_at_points`
- **Does**: Evaluates canonical X/Y checker phase at arbitrary material
  coordinates and extrudes that phase through tissue thickness. A dimensionless
  `1e-6` boundary bias makes exact checker edges stable under rest-pose
  embed/deform floating-point reconstruction.
- **Rationale**: H1 plastic cells must read the same field used to label H0
  reference sites; duplicating parity math would let the contracts drift. A 3D
  parity volume is not a skin texture: depth layers occlude into speckles. For
  this planar scaffold X/Y is the surface chart; later 3D bodies need bone UVs.

### `build_h_body_plan`
- **Does**: Samples capsule volume, assigns top-three soft bone weights, embeds
  exact unclamped axial/side/normal bone coordinates, and measures reference
  kernel density.
- **Rationale**: This privileged niche map is the H0 upper-bound control. H1
  replaces unique assignments with continuous demand, and H2 adds nonuniform
  regional demand without exposing exact target sites.

### `embed_points`
- **Does**: Convert arbitrary world points into top-three soft bone coordinates
  in a specified scaffold frame.
- **Interacts with**: Continuous H1 differentiation and attachment refresh.

### `deform_embedded` / `deform_body_plan`
- **Does**: Apply linear blend deformation to tissue niches and gap probes.
- **Interacts with**: Particle dynamics and metrics.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `particles.py` | Stable site count, identities, checker coordinate contract, and public embedding | Field semantics |
| `dynamics.py` | Deformed sites correspond index-for-index with assignments | Site ordering |
| `metrics.py` | Two deformed gap probes | Gap order/count |
