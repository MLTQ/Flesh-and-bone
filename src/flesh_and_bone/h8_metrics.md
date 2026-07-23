# `h8_metrics.py`

## Purpose

Measure H8's cold-start teacher, finite streaming rollouts, final-hold settling,
and conditional density causality. Periodic seam/cycle-drift metrics are
deliberately absent.

## Components

### `measure_h8_teacher`

- **Does:** Reports amplitude, far/near locality, edge coherence, density bounds,
  hold-entry/final velocity, and finite state for the explicit teacher.

### `measure_h8_rollout`

- **Does:** Compares one backbone/hybrid stream to the teacher with inherited
  RMS/p99/max/amplitude metrics plus final-state and velocity-decay measures.
- **Interacts with:** `H8StreamingRollout` and `H8TeacherTrajectory` from
  `h8_streaming.py`.

### `measure_h8_compression_error`

- **Does:** Recomputes five-percent excess compression on every eighth directed
  edge for each finite frame, without a cycle dimension.

### `acceptance_h8`

- **Does:** Applies frozen safety gates to every variant and causal reduction
  gates only when density and backbone error are non-vacuous.
- **Softness gate:** Requires the hybrid far/near deformation ratio to remain
  within ±10% of the explicit teacher's ratio. This tests spatial-profile
  preservation without demanding a profile that the target trajectory lacks.
- **Rationale:** Gentle motions remain valid ecological stability evidence but
  cannot be counted as proof that density mechanics matter.

### `aggregate_h8`

- **Does:** Requires universal safety, all eligible causal passes, and a chosen
  minimum number of eligible variants.

## Contracts

| Dependent | Expects | Breaking changes |
| --- | --- | --- |
| `h8_experiment.py` | stable named metric/gate keys and JSON-safe values | Key/status changes |
| `experiments/H8.md` | thresholds encode the table plus its logged pre-final amendment | Unlogged threshold changes |
| tests | low-excitation variants skip only causal evidence, not safety | Eligibility semantics |

## Notes

The final velocity limit is `max(35% × hold-entry velocity, 5 mm/s)`, which
avoids unstable ratios when a clip enters the hold nearly stationary.
