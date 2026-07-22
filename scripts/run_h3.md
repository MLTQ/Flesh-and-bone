# run_h3.py

## Purpose

Provides the reproducible CLI for the H3 oracle, learned-fate primary,
local-deficit control, and shortage-blind learned control.

## Components

### `main`
- **Does**: Parses seed/device/arm/output, automatically runs the same-seed
  oracle before any arm requiring an oracle-proximity gate, and writes each arm
  to a separate evidence directory.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Experiment commands | `--arm all` runs one oracle plus three comparisons | CLI choices |
| Learned acceptance | Same-seed oracle pre-wound coverage is supplied | Run ordering |
