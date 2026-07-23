# `index.html`

## Purpose

Provide the semantic, accessible shell for motion composition, progress,
retarget inspection, and clip history. The page is framework-free and works on
the local research network without public internet access.

## Components

### Composition and progress panels

- **Does:** Collect prompt/seed/generation controls and expose measured pipeline
  stages with an explicit synchronous-generation caveat.

### Retarget review panel

- **Does:** Shows the LBS animation, raw/retargeted artifacts, anatomy metrics,
  map detail, and manual decision controls.
- **Rationale:** A prominent rig-only banner and the preview caption identify
  that H7 learned/local flesh mechanics are not running.

### Clip history

- **Does:** Hosts reloadable job buttons populated by `app.js`.

## Contracts

| Dependent | Expects | Breaking changes |
| --- | --- | --- |
| `app.js` | stable IDs for every dynamic field | Renaming/removing IDs |
| `styles.css` | panel and status class names | Renaming classes |
| reviewer | native labels and controls remain keyboard accessible | Removing labels/semantics |

## Notes

No remote fonts, trackers, CDNs, or assets are used. An empty inline favicon
avoids irrelevant missing-file requests in the development log.
