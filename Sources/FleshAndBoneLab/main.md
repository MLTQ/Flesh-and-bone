# `main.swift`

## Purpose

Dispatch between the interactive AppKit lab and the no-window native compute
benchmark.

## Components

### top-level entry

- **Does:** Runs `--benchmark [frames]`,
  `--population-test [recovery frames]`,
  `--render-test [png] [front|left|back|right] [opacity]`, or
  `--render-benchmark` synchronously, otherwise starts the ordinary dock-visible
  app with `AppDelegate`.
- **Interacts with:** `RuntimeAssets` and `RuntimeBenchmark`.

## Contracts

| Dependent | Expects | Breaking changes |
| --- | --- | --- |
| resource notes/CI | `--benchmark [frames]` prints all profiles | CLI shape |
| population QA | `--population-test [frames]` runs directed overcapacity and wound recovery | CLI shape |
| visual QA | render flags use production simulation/renderer and named views | Alternate path |
| `scripts/make_flesh_app.sh` | no flags launches the app | Default behavior |
