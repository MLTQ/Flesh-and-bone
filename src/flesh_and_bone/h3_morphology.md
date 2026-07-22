# h3_morphology.py

## Purpose

Defines the fine H3 humanoid tissue envelope, explicit extra-skeletal lobes,
variable material splat scale, and right-cheek wound. The envelope is a
developmental reference field, not renderer-only geometry.

## Components

### `H3BodyPlan`
- **Does**: Extends the common body plan with 18 anatomical regions, four
  critical lobe regions, wound mask, bone-distance evidence, and per-site splat
  scale.

### `build_h3_body_plan`
- **Does**: Samples tapered limb capsules plus torso, chest, pelvis, head,
  cheek, buttock, hand, and foot ellipsoids at fine spacing and embeds every
  sample in the 15-bone rest frame.
- **Rationale**: Cheeks and buttocks deliberately place valid tissue away from
  bone segments, distinguishing an anatomical demand field from nearest-bone
  filling. Region scale spans 0.80–1.30 while the H3 base radius remains below
  H2's world-space splat radius.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `h3_experiment.py` | Nonempty 18-region plan and right-cheek wound | Region order/mask |
| H3 metrics | Critical-region indices, bone distances, and splat scales | Annotation semantics |
| Learned allocator | Stable region count/order and target capacities | Region order/count |
| Particle material | `splat_scale` is positive and indexed by reference site | Scale field |
