# skeleton.py

## Purpose

Defines the connected five-bone H scaffold, its bounded procedural animation,
and shared point-to-bone coordinate operations. The common frame supports any
positive bone count so later experiments can introduce humanoid scaffolds.

## Components

### `ScaffoldFrame`
- **Does**: Validates one frame of one or more ordered world-space bone segments
  and exposes its dynamic bone count.
- **Interacts with**: Morphology deformation and rendering.

### `HScaffold.frame`
- **Does**: Produces two degree-three junctions, four articulated arms, one
  bridge, and a bounded global spin/depth jiggle.
- **Rationale**: Motion begins exactly at the rest pose so assembly and moving
  phases share one canonical body plan.

### `segment_projection` / `bone_frames`
- **Does**: Compute clamped capsule distance and stable bone-local coordinates.
- **Interacts with**: `build_h_body_plan` in `morphology.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `morphology.py` | Ordered segments and orthonormal frame columns | Bone order/frame convention |
| H renderer/tests | H frames retain endpoint shape `(5,2,3)` | H bone order |
| H3 humanoid scaffold | Arbitrary endpoint shape `(B,2,3)` is accepted | Generic validation |
