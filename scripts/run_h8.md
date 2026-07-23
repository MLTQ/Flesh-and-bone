# `run_h8.py`

## Purpose

Expose H8's qualification and sealed final stages through an explicit CLI.
Every source is supplied as `NAME=retargeted_motion.npz`; the experiment module
then verifies its embedded prompt, seed, fps, and contact-IK provenance.

## Components

### `_motion_paths`

- **Does:** Parses unique named files and fails before loading mechanics when a
  path is absent or malformed.

### `main`

- **Does:** Constructs the frozen H8 config, runs one stage, and prints the
  aggregate verdict.
- **Interacts with:** `run_h8_stage` in `h8_experiment.py`.

## Contracts

| Dependent | Expects | Breaking changes |
| --- | --- | --- |
| evidence commands | repeated `--motion NAME=PATH` arguments | CLI syntax |
| `h8_experiment.py` | stage is `qualification` or `final` | Stage vocabulary |
| final suite | qualification report exists and passes in same output | Output-directory semantics |

## Usage

Qualification requires exactly `standing_wave` and `two_step`; final requires
exactly `clean_walk`, `side_sway`, and `quick_turn`. CUDA evidence runs should
use `--device cuda` and retain the default output
`experiments/runs/h8_streaming`.
