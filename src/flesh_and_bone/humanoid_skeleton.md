# humanoid_skeleton.py

## Purpose

Defines the first variable-bone-count humanoid precursor and a bounded
walk/sway animation. It is intentionally a coarse rig for developmental tests,
not a production skeletal hierarchy.

## Components

### `HUMANOID_BONE_NAMES`
- **Does**: Fixes the semantic order of the 15 trunk, head, arm, and leg bones.

### `HumanoidScaffold.frame`
- **Does**: Constructs a connected 15-segment frame with exactly opposed limb
  depth swings, small body sway, and an exact zero-time assembly pose.
- **Interacts with**: `ScaffoldFrame` and H3 morphology deformation.
- **Rationale**: Direct joint construction keeps shared endpoints exactly
  connected while providing enough articulation to reveal attachment failure.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `h3_morphology.py` | Stable 15-bone order and zero-time landmarks | Bone order/rest coordinates |
| `h3_experiment.py` | `frame(time, device, dtype)` matches H scaffold API | Method signature |
| Particle attachments | Frames remain shaped `(15, 2, 3)` | Bone count |

## Notes

The animation is deliberately bounded and starts at the rest pose. A future
human rig should load joint transforms and skinning semantics from an external
skeleton rather than expanding this procedural prototype.
