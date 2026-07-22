# render_splat_sweep.py

## Purpose

Separates visual splat coverage from physical cell density by rendering a fixed
validated volume across radius/opacity combinations, including a common face
crop for detail inspection.

## Components

### `main`
- **Does**: Skins phase zero, renders radius scales 0.30-0.60 across opacities
  0.52-0.90, and writes full-body and face contact sheets.
- **Interacts with**: H4 volume/rig loaders, texture sampling, and the splat
  renderer.
- **Rationale**: A denser grid with the same radius-to-pitch ratio retains the
  same fractional holes; density and overlap must be selected independently.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Dense experiment | Four columns ordered by radius, three rows by opacity | Sweep ordering |
| Visual review | Every tile labels exact radius and opacity | Label removal |
