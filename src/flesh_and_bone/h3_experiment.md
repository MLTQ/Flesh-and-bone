# h3_experiment.py

## Purpose

Runs H3's fine humanoid assembly, learned coarse-fate allocation, remote cheek
damage/re-feeding, local maturation, and articulated motion protocol for the
primary and causal-control arms.

## Components

### `H3Config`
- **Does**: Records arm, deterministic training/rollout seed, phase boundaries,
  fine spacing, small render radius, feed source, learner budget, and oracle
  comparison value. The frozen H3 default uses deficit log weight 2.8; seed-7
  tuning showed that a longer repair window did not improve the missed cheek
  sample, while stronger within-region deficit contrast did.

### `run_h3`
- **Does**: Builds the 15-bone/18-region body, optionally trains and freezes the
  fate MLP, grows generation 0, deletes the right-cheek wound, feeds generation
  1, locally matures it, articulates the body, and writes metrics/model/media.
- **Interacts with**: `HumanoidScaffold`, `H3BodyPlan`, `DeficitDynamics`, fate
  model, H3 metrics, and the generic variable-radius renderer.

### H3 arms
- **Oracle**: Uses the H2 capacity-and-distance selector as a mechanical upper
  bound.
- **Learned**: Uses the frozen distilled selector with live capacity.
- **Local deficit**: Removes coarse guides entirely.
- **No shortage**: Uses identical learned weights while hiding capacity and
  availability inputs.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `scripts/run_h3.py` | `run_h3(path, config)` returns a serializable report | Signature/config |
| Experiment ledger | Oracle is run before learned proximity is judged | Baseline semantics |
| Lineage metrics | Damage is measured before generation-1 feeding | Phase ordering |
| Reproducibility | Model and rollout share recorded seed but separate generators | Seed behavior |
