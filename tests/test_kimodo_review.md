# `test_kimodo_review.py`

## Purpose

Test the HTTP client contract, public request bounds, and durable decision state
without invoking live Kimodo, Archepelago, dense skinning, or GIF rendering.

## Components

### Temporary Kimodo stub

- **Does:** Records exact generation JSON and returns an in-memory NPZ response.
- **Interacts with:** `KimodoGenerationClient` in `kimodo_client.py`.

### Validation and persistence tests

- **Does:** Covers random seeds, duration/step limits, manifest rediscovery, and
  accepted/rejected note persistence.

## Contracts

| Dependent | Expects | Breaking changes |
| --- | --- | --- |
| `kimodo_client.py` | payload includes prompt, duration, seed, steps, postprocess | API payload changes |
| `ReviewJobManager` | completed manifests reload and decisions rewrite them | Job/manifest schema changes |

## Notes

The expensive live path is exercised as an end-to-end console validation and
recorded in `experiments/KIMODO_REVIEW.md`.
