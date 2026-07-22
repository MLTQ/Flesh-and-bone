# fate_model.py

## Purpose

Defines and trains H3's shared coarse anatomical-fate scorer. It distills the
frozen H2 capacity-and-distance rule while remaining permutation-equivariant
over the number and order of regions.

## Components

### `RegionalFateMLP`
- **Does**: Applies one small MLP independently to each region's shortage,
  availability, and normalized distance, producing comparable region scores.
- **Rationale**: Shared scoring avoids an output head tied to exactly 18 H3
  regions and prevents region identity from becoming an implicit lookup table.

### `oracle_fate_scores`
- **Does**: Expresses the deterministic teacher as a continuous score target;
  unavailable regions receive a large penalty.

### `train_fate_model`
- **Does**: Trains on deterministic randomized capacity/distance states and
  reports loss plus held-out argmax agreement before any morphology rollout.
  The seed is applied before model construction as well as to the sample
  generator, making initialization and examples deterministic.
- **Rationale**: A separate holdout makes allocator failure distinguishable
  from particle-dynamics failure.

### `LearnedFateSelector`
- **Does**: Converts a frozen model into the scalar regional-choice callback
  used when new particles acquire a hysteretic guide. The ablation can hide
  shortage and availability without changing model weights.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `DeficitDynamics` | Selector accepts shortage, capacity, and distance | Callback signature |
| H3 runner | Training is deterministic and returns model plus serializable report | Return contract |
| H3 metrics/ledger | Held-out agreement is measured before rollout | Report fields |

## Notes

This is supervised oracle distillation. It learns fate scoring but still
receives a global regional-capacity vector; it is not evidence of purely local
development or end-to-end neural morphogenesis.
