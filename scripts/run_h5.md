# run_h5.py

## Purpose

Provides the reproducible CLI for H5 teacher construction, three-seed local
rule training, free rollout, ablation, checkpointing, and visuals.

## Components

### `main`
- **Does**: Parses output/device/seeds, runs H5, and prints aggregate plus
  per-seed one-step/rollout results.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Experiment commands | Default seeds are 7, 19, and 31 | CLI/defaults |
| GPU runner | `--device cuda` keeps training/rollout tensors on GPU | Device behavior |
