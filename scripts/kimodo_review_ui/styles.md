# `styles.css`

## Purpose

Present the console as a dense research instrument across desktop and mobile.
Acid green denotes active/pass, amber warning, red failure, and blue downloadable
evidence.

## Components

### Layout system

- **Does:** Uses a two-column compose/progress workspace with full-width review
  and history panels; collapses at 980 and 650 pixels.

### Status and result components

- **Does:** Styles progress stages, verdict badges, metric cards, preview frame,
  stacked rig-only mode banner, decisions, and job history without relying on
  color alone.

### Accessibility rules

- **Does:** Adds high-contrast focus treatment and disables animation under
  `prefers-reduced-motion`.

## Contracts

| Dependent | Expects | Breaking changes |
| --- | --- | --- |
| `index.html` | CSS variables and declared component classes | Class/variable removal |
| `app.js` | `pass`, `warn`, `fail`, job-status, and `hidden` classes | Status-class semantics |
| mobile reviewer | single-column controls remain usable below 650 px | Breakpoint/layout changes |

## Notes

The stylesheet imports no fonts or images; gradients, borders, and typography
provide the entire visual system.
