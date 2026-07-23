# `kimodo_client.py`

## Purpose

Own the external Kimodo boundary: configuration, request validation, health
checks, and generation calls. Keeping HTTP concerns separate lets the retarget
pipeline and job queue be tested without a live generation service.

## Components

### `KimodoReviewConfig`

- **Does:** Defines service, asset, output, and bounded-preview settings.
- **Interacts with:** `ReviewJobManager` in `kimodo_review.py` and
  `render_review_preview` in `kimodo_preview.py`.
- **Rationale:** Relative repository assets are resolved once at manager startup;
  Archepelago defaults to `~/Code/Archepelago` and remains overridable.

### `KimodoGenerationClient`

- **Does:** Calls synchronous `/healthz` and `/generate` routes with `urllib`.
- **Interacts with:** `run_review_job` in `kimodo_review.py`.
- **Rationale:** Returning raw response bytes preserves the server artifact
  exactly and avoids adding an HTTP dependency.

### `validate_review_request`

- **Does:** Normalizes prompts, seeds, generation controls, and contact-IK state.
- **Interacts with:** Both the HTTP API and job manager submission path.

## Contracts

| Dependent | Expects | Breaking changes |
| --- | --- | --- |
| `run_kimodo_review.py` | `KimodoReviewConfig` accepts CLI overrides | Renaming fields or changing path resolution |
| `kimodo_review.py` | client returns NPZ bytes and health JSON | Returning parsed/mutated motion instead of bytes |
| `app.js` via API | validation bounds match form controls | Changing prompt, duration, seed, or step bounds |

## Notes

Kimodo exposes no sampling-progress endpoint. The client does not invent one;
the browser labels its within-generation percentage as an estimate.
