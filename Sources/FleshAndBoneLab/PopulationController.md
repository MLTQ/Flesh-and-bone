# `PopulationController.swift`

## Purpose

Define layered per-niche source/vacuum population state independently from
Metal encoding, brush selection, and UI presentation.

## Components

### `PopulationChange`

- **Does:** Carries arbitrary GPU slot indices selected by one directed brush
  stroke and whether those slots become active or dormant.
- **Interacts with:** `FleshSimulation.encodePopulationChange`.

### `PopulationController`

- **Does:** Tracks 0/1/2 occupancy for every anatomical niche. Source activates
  the first free layer; vacuum removes the highest active layer.
- **Rationale:** Separate strokes can paint a region from empty through the
  100% baseline to 200% without changing untouched anatomy.

## Contracts

| Dependent | Expects | Breaking changes |
| --- | --- | --- |
| `FleshSimulation` | layer zero starts full, layer one dormant | Slot mapping |
| `PopulationBrush` | template eligibility and representative active slots are queryable | Occupancy API |
| `ControlPanel` | active count is reported relative to `baseCount`, not capacity | Percentage semantics |

## Notes

Capacity is preallocated at 200%. Layer-two cells clone a niche's target,
material, and skin weights. This permits targeted overpopulation but remains an
explicit mechanical stress test, not learned fate choice or unbounded growth.
