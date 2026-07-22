# run_h1.py

## Purpose

Runs the continuous-deficit H1 arm, legacy first-contact texture control,
pressure-off control, nearest-bone control, or all arms with identical seeds and
schedules.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| README/users | `--arm main|legacy_texture|pressure_off|nearest_bone|all` | CLI flags/arm names |
| Experiment ledger | Each output directory names arm and seed | Output naming |
