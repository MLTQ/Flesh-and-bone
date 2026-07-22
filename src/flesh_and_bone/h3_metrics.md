# h3_metrics.py

## Purpose

Measures H3's fine humanoid representation, extra-skeletal lobe coverage,
learned guide allocation, cheek repair, and articulated persistence separately
from the controller loss.

## Components

### `measure_h3`
- **Does**: Extends H2 region/lineage evidence with mean cheek-and-buttock
  coverage, extra-skeletal coverage, guide-allocation error, and active material
  splat-scale range.

### `representation_gates`
- **Does**: Verifies bone/region count, fine spacing, world-space splat size,
  variable scale, and the fraction of reference tissue meaningfully displaced
  from bones.

### `acceptance_h3_oracle`
- **Does**: Applies the predeclared mechanical upper-bound development and
  motion gates.

### `acceptance_h3_learned`
- **Does**: Adds held-out learner agreement, proximity to the oracle rollout,
  remote cheek repair, full learned-guide coverage, lineage, mass, and
  assignment-leakage gates.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `h3_experiment.py` | Stable representation, four-phase, and acceptance keys | Signature/key changes |
| Experiment ledger | Thresholds exactly match predeclared H3 protocol | Gate changes |
| H3 controls | Metrics remain meaningful when no cells receive guides | Empty-guide semantics |
