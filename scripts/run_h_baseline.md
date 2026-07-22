# run_h_baseline.py

## Purpose

Provides the reproducible command-line interface for H0, with explicit device,
seed, duration, motion onset, capture cadence, and output directory.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| README/users | `python scripts/run_h_baseline.py --device ...` | CLI flags |
| `experiment.py` | Validated `ExperimentConfig` passed to `run_h0` | Config mapping |
