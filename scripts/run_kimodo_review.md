# `run_kimodo_review.py`

## Purpose

Serve the static Kimodo review console and expose the job manager through a
small local JSON/artifact API. It uses the standard library so the research UI
adds no web-framework dependency.

## Components

### `ReviewHandler`

- **Does:** Routes health, job submission/polling, decisions, static assets, and
  run artifacts.
- **Interacts with:** `ReviewJobManager` in `kimodo_review.py` and the files in
  `scripts/kimodo_review_ui`.
- **Rationale:** Artifact resolution is confined beneath the configured ignored
  output directory; request bodies are size-bounded.

### `main`

- **Does:** Resolves CLI overrides, creates the single-worker manager, and runs a
  `ThreadingHTTPServer` on `127.0.0.1:8787` by default.
- **Rationale:** Threaded HTTP keeps progress polling responsive during a
  synchronous generation/render job.

## Contracts

| Dependent | Expects | Breaking changes |
| --- | --- | --- |
| `app.js` | `/api/health`, `/api/jobs`, decision, and artifact routes | Route or JSON error schema changes |
| local user | default URL is `http://127.0.0.1:8787` | Host/port defaults |
| manager | Kimodo defaults to `192.168.0.202:8111`, Archepelago to `~/Code/Archepelago` | CLI/config mapping |

## Usage

```bash
PYTHONPATH=src python scripts/run_kimodo_review.py
```

Use `--kimodo-url`, `--archepelago`, `--preview-size`, `--preview-frames`, or
`--preview-cells` when the service or review budget changes.
