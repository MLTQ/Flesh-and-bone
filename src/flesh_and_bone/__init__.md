# __init__.py

## Purpose

Exposes the small public surface needed to construct either the original H or
H3 humanoid scaffold, derive their body plans, and simulate a particle
reservoir.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| Experiment scripts | Named scaffold, body-plan, and particle types import here | Export removal/rename |
| External prototypes | H and humanoid scaffold/body-plan constructors are stable public entries | Export semantics |
