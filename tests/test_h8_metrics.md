# `test_h8_metrics.py`

## Purpose

Encode H8's distinction between universal streaming safety and conditional
causal evidence without allocating production trajectories.

## Components

### Per-variant acceptance tests

- **Does:** Confirms an eligible motion must clear both reduction gates, while a
  low-density motion can pass only as ecological safety evidence.
- **Does:** Confirms the softness gate preserves the teacher's spatial profile,
  including valid far/near ratios close to one, and rejects distorted profiles.

### Aggregate eligibility test

- **Does:** Confirms the sealed final suite needs at least three causally
  eligible clip/timing variants in addition to universal safety.

## Contracts

| Dependent | Expects | Breaking changes |
| --- | --- | --- |
| `h8_metrics.py` | eligibility threshold and safety/causal separation | Gate semantics |
| `experiments/H8.md` | final causal floor equals three variants | Aggregate threshold |
