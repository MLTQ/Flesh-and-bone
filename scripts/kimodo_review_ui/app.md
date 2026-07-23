# `app.js`

## Purpose

Drive the static console against the local JSON API: submit prompts, poll one
active job, restore history, render diagnostics, and persist manual decisions.

## Components

### API and polling functions

- **Does:** Wrap `fetch`, select jobs, estimate only the synchronous Kimodo
  portion, and stop polling at completion/failure.
- **Interacts with:** `ReviewHandler` in `run_kimodo_review.py`.
- **Rationale:** Generation eases only from 10% toward 46%; later boundaries are
  real backend stages and the UI labels the estimate.

### Result renderers

- **Does:** Convert metric units, populate artifact links, display semantic-map
  context, and restore decisions/history.
- **Interacts with:** The manifest schema from `kimodo_review.py`.

### Event handlers

- **Does:** Bind prompt chips, generation form, history selection, and
  accept/reject controls after `DOMContentLoaded`.

## Contracts

| Dependent | Expects | Breaking changes |
| --- | --- | --- |
| `index.html` | all referenced element IDs exist once | Renaming IDs or form controls |
| backend | generation and decision JSON use documented fields | Payload key changes |
| reviewer | HTML derived from prompts/manifests is escaped | Removing escaping before `innerHTML` |

## Notes

History reloads from disk-backed manager state. Selecting a running job resumes
polling; selecting a complete one restores artifacts, metrics, and notes.
