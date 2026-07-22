# particles.py

## Purpose

Owns the fixed-capacity Gaussian-cell reservoir, H0 niche reservations, H1
unassigned feeding, persistent bone coordinates, per-material splat scale, and
two-stage differentiation across variable-bone-count scaffolds.

## Components

### `ParticleSystem`
- **Does**: Stores position, velocity, mass, active/committed flags, site and
  tissue identities, material-lock/stability state, birth generation, soft bone
  weights/local coordinates, a coarse developmental guide region, material
  splat scale, plus recurrent channels. `bone_count` defaults to five for H0–H2
  and is explicit for later scaffolds.

### `feed`
- **Does**: Activates reservoir cells at an injection point and reserves unique,
  nearest currently unfilled developmental niches.
- **Rationale**: Unique reservations are the privileged H0 control and establish
  an upper bound before continuous deficit-following replaces them.

### `update_commitment`
- **Does**: Assigns persistent H0 body-part and checker identity after arrival.
- **Rationale**: Identity cannot be recomputed from world position without
  hiding material sliding during motion.

### `feed_unassigned` / `update_continuous_commitment`
- **Does**: Activate H1 cells with no target index, then differentiate them from
  proximity to the continuous moving body field. Contacted H1 cells remain
  materially plastic until explicitly locked.

### `refresh_continuous_attachments` / `attachment_targets`
- **Does**: Capture migrating material in bone coordinates before animation and
  deform those persistent coordinates afterward.

### `refresh_plastic_material` / `lock_material`
- **Does**: Let contacted H1 cells read checker phenotype from the nearest local
  sample of the body material field during assembly, including optional splat
  scale, then irreversibly lock phenotype and attachment state before
  articulation. The sample index is never stored or reserved.
- **Rationale**: First-contact color is stale after continued migration. A
  plastic-to-locked transition separates assembly from persistent material
  identity without assigning cells a target niche.

### `update_local_maturation`
- **Does**: Accumulates consecutive locally stable steps and locks eligible
  cells after a configured duration.
- **Interacts with**: H2 repair dynamics, where replacement cells arrive after
  the initial global developmental stage.

### `remove`
- **Does**: Deactivates explicitly selected cells, clears their phenotype and
  recurrent state, and returns their slots to the feedable reservoir.
- **Rationale**: Wound experiments require recoverable, mass-accounted deletion
  rather than teleporting cells away or retaining hidden state.

### `guide_region`
- **Does**: Stores an optional coarse anatomical commitment for H2 transport;
  `-1` means no region has been selected.
- **Rationale**: A hysteretic region guide prevents long-range deficit targets
  from chattering while still withholding any final particle/site position.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `dynamics.py` | H0 cells have sites; H1 cells use `-1` plus bone attachments | Assignment/attachment semantics |
| `metrics.py` / `h2_metrics.py` | Mass, generation, guide, maturation, identity, and lock fields | State layout |
| `render.py` | Uncommitted identity is `-1`; checker is 0/1 after commitment; splat scale is positive | Sentinel/labels/scale |
| H3 humanoid | Bone-state tensors match the configured scaffold count | Constructor bone count |
