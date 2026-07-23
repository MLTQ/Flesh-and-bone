# `kimodo_preview.py`

## Purpose

Render a bounded visual review of a retargeted clip without running learned
flesh mechanics. The module owns only dense-cell sampling, chunked skinning, and
the three preview image artifacts.

## Components

### `render_review_preview`

- **Does:** Writes `character.gif`, `contact_sheet.png`, and
  `anatomy_frame.png`, then returns JSON-safe render metadata.
- **Interacts with:** H4 texture rendering, H5U volume state,
  `hierarchy_segments`, and `bone_group_colors`.
- **Rationale:** Root-following X/Z framing keeps locomotion visible while world
  Y reveals jumps or vertical drift.

### `_skin_selected`

- **Does:** Skins bounded frames/cells in small chunks.
- **Interacts with:** `linear_skin` in `rig_asset.py`.
- **Rationale:** Avoids allocating the full `[frames,bones,cells,4]` temporary.

### `_contact_sheet`

- **Does:** Places up to twelve evenly sampled GIF frames in a 4-column sheet.

## Contracts

| Dependent | Expects | Breaking changes |
| --- | --- | --- |
| `kimodo_review.py` | returned metadata is JSON-safe and all three images exist | Artifact names or return keys |
| `app.js` | square GIF and diagnostic URLs resolve after job completion | Renaming output files |
| reviewer | preview is pure LBS and uses 22 core hierarchy joints | Adding learned flesh or helper-bone segments |

## Notes

Sampling changes preview cost, not the stored 180-frame retargeted artifact.
The default uses at most 24,000 of 91,979 cells and 45 source frames.
