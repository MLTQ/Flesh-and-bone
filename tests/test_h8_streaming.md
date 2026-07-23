# `test_h8_streaming.py`

## Purpose

Verify H8's finite-time boundary conditions and cold-start integration on a
three-cell directed graph, without loading production assets or checkpoints.

## Components

### Resampling and acceleration tests

- **Does:** Confirms 2x resampling preserves both endpoints, final holds repeat
  only the last pose, and acceleration never wraps last-to-first.

### Exact-backbone streaming test

- **Does:** Starts teacher and rollout at exact zero state and verifies a
  constitutive rule matching the small explicit teacher reproduces its complete
  finite stream.

## Contracts

| Dependent | Expects | Breaking changes |
| --- | --- | --- |
| `h8_streaming.py` | visible states are captured before each frame's substeps | Capture timing |
| H8 protocol | no periodic wrap or warmed teacher initialization | Boundary/initial state changes |
| test body | graph exposes source, target, and degree | Graph contract changes |
