# `kimodo_diagnostics.py`

## Purpose

Provide pure NumPy anatomy checks for retargeted skeletons. The thresholds
detect gross convention/topology failures without acting as a restrictive pose
prior for athletic or stylized motion.

## Components

### `AnatomyThresholds`

- **Does:** Declares warning/failure bounds for pelvis tilt, raised hip sockets,
  hierarchy drift, and planted-foot drift.

### `analyze_retarget`

- **Does:** Returns a JSON-safe verdict, metric rows, contextual joint angles,
  worst frame, and per-frame pelvis traces for endpoints shaped
  `[frames,bones,2,3]`.
- **Interacts with:** `run_review_job` in `kimodo_review.py` and metric cards in
  `app.js`.
- **Rationale:** Knee/elbow extremes are context only because legitimate motion
  can bend or cross limbs.

### `hierarchy_segments`

- **Does:** Joins parent/child heads for debug rendering.
- **Rationale:** `include_count=22` excludes two unweighted Blender head helpers.

### `bone_group_colors`

- **Does:** Assigns stable bilateral/torso/head colors from dominant skin bones.
- **Rationale:** A leg attached to an arm or torso becomes visually immediate.

## Contracts

| Dependent | Expects | Breaking changes |
| --- | --- | --- |
| `kimodo_review.py` | verdict and metrics are JSON-safe | Return schema or status labels |
| `kimodo_preview.py` | segments accept single/all-frame endpoints | Shape semantics |
| UI | units are `m`, `deg`, or `bool` | Unit/status vocabulary |

## Notes

Contact drift uses Kimodo's 77-joint heel channels `(0, 3)`, ignores runs shorter
than three frames, and reports unavailable rather than passing when no span
exists. The final three frames are excluded because Archepelago's contact IK
deliberately eases out over that release window.
