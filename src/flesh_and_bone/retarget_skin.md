# retarget_skin.py

## Purpose

Bridges a canonical rest-relative animation (such as Archepelago's Kimodo
retarget output) into the existing H4 rig's homogeneous skin matrices and bone
endpoint evidence without Blender.

## Components

### `canonical_motion_skin`
- **Does**: Accumulates canonical local rotations over the H4 hierarchy,
  reconstructs posed joint locations from rest-head offsets and animated root,
  composes each world rotation with its stored bind basis, and returns
  `pose @ inverse(bind)` skin matrices plus posed head/tail segments.
- **Rationale**: Identity local rotations at the rest root must produce identity
  skin matrices exactly; that closed check prevents coordinate mistakes from
  being blamed on flesh mechanics.

### `palindrome_close`
- **Does**: Forms a periodic forward/return cycle as `0..N-1,N-2..1`, avoiding
  duplicate endpoints while making the wrap continue from frame one to zero.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Kimodo preparation | Local rotations are canonical and rest-relative | Rotation convention |
| H4 skinning | Asset bind matrices map bone-local coordinates into canonical rest space | Bind semantics |
| Periodic teacher | Palindrome output has `2N-2` phases and no duplicate endpoints | Closure indexing |
