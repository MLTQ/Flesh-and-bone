# run_h4.py

## Purpose

Provides the ordinary-Python CLI for H4 after Blender extraction has produced
the canonical rig bundle.

## Components

### `main`
- **Does**: Accepts output/archive/rig/pitch overrides, runs the complete H4
  evidence pipeline, and prints its verdict plus selected influence count and
  volume size.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Experiment commands | Defaults reference the checked-in source zip and derived rig | Default paths |
| H4 automation | Process exits normally even for a metric FAIL and records details | CLI behavior |
