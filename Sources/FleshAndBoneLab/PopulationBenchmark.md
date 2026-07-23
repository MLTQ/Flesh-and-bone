# `PopulationBenchmark.swift`

## Purpose

Exercise the exact directed native source/vacuum path without UI timing or
visual interpretation. Verify both local overcapacity and local wound recovery
against the reference population.

## Components

### `PopulationBenchmark.run`

- **Does:** Uses the production front-surface brush to activate reserve cells
  in a centered patch, rolls that overcapacity forward, removes the reserve,
  vacuums the same baseline cells, dwells for 30 frames, and refills them.
- **Reports:** Selected niche count, peak and wound population relative to the
  100% body, source/recovered residual RMS, paired-layer separation, exact
  restoration, finite state, and a conservative verdict.
- **Causal control:** Repeats the overcapacity rollout with only the H6C
  backbone. A pass requires the H7C cross-layer density observation to produce
  more paired separation than this otherwise identical control.
- **Rationale:** Reads the real shared Metal buffers after completed command
  buffers; controller counters alone cannot prove the GPU mutation occurred.

## Contracts

| Dependent | Expects | Breaking changes |
| --- | --- | --- |
| `main.swift` | `--population-test [recovery frames]` invokes all profiles | CLI changes |
| population controller | occupancy transitions are 0↔1↔2 per niche | Layer semantics |
| experiment notes | pass requires exact counts, finite state, contracted source RMS, and nonzero pair separation | Verdict changes |

## Notes

This is a transport/re-feeding and local-density stress test, not autonomous
development. Every cell retains an exported niche, material, weights, and
same-layer neighbors.
