# run_h2.py

## Purpose

Runs the H2 main arm, nearest-bone/uniform-density control, pressure-off control,
first-contact-material control, or every arm with identical schedules.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| README/users | `--arm main|nearest_bone|pressure_off|first_contact|all` | CLI flags/arm names |
| Experiment ledger | Output directory names arm and seed | Output naming |
