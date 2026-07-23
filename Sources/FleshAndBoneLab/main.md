# `main.swift`

## Purpose

Dispatch between the interactive AppKit lab and the no-window native compute
benchmark.

## Components

### top-level entry

- **Does:** Runs `--benchmark [frames]`, `--render-test [png]`, or
  `--render-benchmark` synchronously, otherwise starts the ordinary dock-visible
  app with `AppDelegate`.
- **Interacts with:** `RuntimeAssets` and `RuntimeBenchmark`.

## Contracts

| Dependent | Expects | Breaking changes |
| --- | --- | --- |
| resource notes/CI | `--benchmark [frames]` prints all profiles | CLI shape |
| visual QA | render flags use the production simulation/renderer | Alternate path |
| `scripts/make_flesh_app.sh` | no flags launches the app | Default behavior |
