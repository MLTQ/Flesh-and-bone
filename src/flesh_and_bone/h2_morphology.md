# h2_morphology.py

## Purpose

Defines the H2 asymmetric tissue envelope and its predeclared wound patch. The
shape deliberately cannot be represented as a uniform nearest-bone offset.

## Components

### `H2BodyPlan`
- **Does**: Extends the common body plan with anatomical region labels, wound
  samples, target bone-distance profile, and a uniform-density control value.

### `build_h2_body_plan`
- **Does**: Samples tapered per-bone capsules, a terminal upper-left bulb, and
  an off-axis right-junction pad; embeds them in the common five-bone scaffold.
- **Rationale**: The thin bridge, unequal limbs, bulb, and pad distinguish a
  true tissue-demand field from nearest-bone attraction plus uniform pressure.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `h2_experiment.py` | Seven stable regions and a nonempty terminal-bulb wound | Region order/mask semantics |
| `h2_metrics.py` | Region, wound, bone-distance, and uniform-density fields | Field removal/type |
| `DeficitDynamics` | Common `BodyPlan` geometry/density interface | Base-field semantics |
