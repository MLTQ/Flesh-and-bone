# flesh_rule.py

## Purpose

Defines H5's shared local message features and normalized residual-acceleration
MLP. The rule contains no cell index, bone identity, phase, or final position.

## Components

### `flesh_features`
- **Does**: Concatenates residual/velocity, LBS acceleration, neighbor residual/
  velocity differences, bone distance, and local stiffness into 17 raw channels.

### `FleshResidualRule`
- **Does**: Stores fixed feature/target normalization and maps every cell through
  one 96-channel SiLU MLP to raw residual acceleration.
- **Rationale**: Dataset scaling belongs in the checkpoint so rollout and
  training cannot disagree; normalization is buffered, not learned.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| H5 trainer | Exactly 17 feature channels and 3 acceleration outputs | Feature order/count |
| H5 rollout | `forward` accepts raw features and returns physical units | Normalization semantics |
| Checkpoint | Normalization buffers are included in `state_dict` | Buffer removal |
