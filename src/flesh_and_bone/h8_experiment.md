# `h8_experiment.py`

## Purpose

Orchestrate H8's qualification and sealed final motion suites using frozen H6C
and H7C checkpoints. It validates prompt/seed provenance, skins full retargeted
clips, evaluates both timing arms, persists metrics, and stores bounded render
subsets.

## Components

### `QUALIFICATION_SPECS` / `FINAL_SPECS`

- **Does:** Freeze exact motion names, prompts, and seeds before output is opened.
- **Rationale:** Final semantic holdouts cannot be regenerated opportunistically
  after seeing mechanics results.

### `H8Config`

- **Does:** Freezes source assets, checkpoints, speeds, hold duration, checkpoint
  seeds, diagnostic sampling, and render bounds.

### `load_h8_lbs_motion`

- **Does:** Loads portable retarget rotations/root motion and skins the complete
  91,979-cell body in four-frame chunks.
- **Interacts with:** `canonical_motion_skin` and `linear_skin`.

### `run_h8_stage`

- **Does:** Enforces exact qualification/final source sets, refuses final access
  before qualification passes, and aggregates universal safety plus conditional
  causal evidence.
- **Safety aggregation:** Keeps universal safety separate from conditional causal
  benefit so a causally ineligible arm cannot be mislabeled as mechanically
  unsafe.
- **Interacts with:** Streaming functions in `h8_streaming.py`, metrics in
  `h8_metrics.py`, and frozen loaders from `h7_experiment.py`.

### `_render_state`

- **Does:** Saves at most 60 frames and every fourth cell for seed-7 visual
  evidence without persisting multi-gigabyte full trajectories.

## Contracts

| Dependent | Expects | Breaking changes |
| --- | --- | --- |
| `scripts/run_h8.py` | exact stage/motion mapping and report paths | CLI/stage schema changes |
| H8 renderer | `flesh-and-bone-h8-render-v1` tensors and metadata | Render-state names/shapes |
| `experiments/H8.md` | final is sealed until qualification pass | Removing provenance/seal checks |
| review artifacts | metadata contains exact prompt, seed, 30 fps, contact IK | Weakening source validation |

## Notes

Backbone metrics are computed once per variant because every checkpoint shares
the same frozen H6C rule. Full hybrid trajectories are released after metrics;
only bounded visual subsets and JSON reports remain on disk.
