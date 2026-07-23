# `kimodo_review.py`

## Purpose

Orchestrate one reproducible Kimodo-to-character review job and serialize those
jobs through a durable, pollable worker queue. HTTP transport and preview
rendering live in dedicated modules so this file remains focused on retargeting
and job lifecycle.

## Components

### `run_review_job`

- **Does:** Preserves raw bytes, imports Archepelago's SOMA tools, retargets the
  full sequence, optionally applies contact IK, diagnoses the skeleton, stores a
  portable motion NPZ, invokes preview rendering, and writes `manifest.json`.
- **Interacts with:** `KimodoGenerationClient` in `kimodo_client.py`,
  `analyze_retarget` in `kimodo_diagnostics.py`, `render_review_preview` in
  `kimodo_preview.py`, and the canonical H4 skin bridge.
- **Rationale:** The full source sequence is retained; no H6K truncation,
  palindrome closure, or learned flesh mechanics is introduced.

### `ReviewJobManager`

- **Does:** Maintains one background worker, thread-safe job state, manifest
  rediscovery, and persisted manual decisions.
- **Interacts with:** `ReviewHandler` in `scripts/run_kimodo_review.py`.
- **Rationale:** Kimodo is single-flight, while UI polling must remain responsive.

### Archepelago adapters

- **Does:** Build the 24-bone destination rig and measure raw SOMA FK agreement.
- **Interacts with:** The checked-in Archepelago role profiler, retargeter,
  contact IK, and SOMA skeleton under `~/Code/Archepelago/backend/motion`.

## Contracts

| Dependent | Expects | Breaking changes |
| --- | --- | --- |
| `run_kimodo_review.py` | manager exposes `submit`, `get`, `list`, and `decide` | Method names, job schema, or decision states |
| `app.js` via HTTP API | stages, progress, result, and error remain JSON-safe | Renaming job/status fields |
| reviewer | raw and retargeted NPZ plus manifest survive restarts | Artifact format/name changes |
| `kimodo_review.py` tests | client/config/validator remain imported here for compatibility | Removing those module-level imports |

## Notes

Runtime output is ignored under `experiments/runs/kimodo_review`. A manual
decision mutates only the manifest; it does not launch training or copy a clip
into an experiment.
