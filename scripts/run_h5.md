# run_h5.py

## Purpose

Provides the reproducible CLI for H5 teacher construction, three-seed local
rule training, free rollout, ablation, checkpointing, and visuals.

## Components

### `main`
- **Does**: Parses output/device/seeds plus optional volume, optimizer budget,
  batch size, graph-coupling, image size, splat-overlap, and opacity overrides;
  runs H5; and prints aggregate plus per-seed one-step/rollout results.
- **Rationale**: Density-scaling trials need to select a separately frozen
  volume and preserve continuum stiffness without editing source defaults.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Experiment commands | Default seeds are 7, 19, and 31 | CLI/defaults |
| GPU runner | `--device cuda` keeps training/rollout tensors on GPU | Device behavior |
| Density trials | Overrides are serialized by `run_h5` in resolved config | Argument forwarding |
| Render sweeps | Radius is a multiple of pitch; opacity is explicit | Render semantics |
