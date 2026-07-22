# dynamics.py

## Purpose

Implements the H0 mechanical upper-bound controller and defines the shared local
particle-NCA residual interface that H1 will train.

## Components

### `DynamicsConfig`
- **Does**: Names integration, attraction, damping, pressure, repulsion,
  commitment, speed, and learned-residual scales.

### `ParticleNCARule`
- **Does**: Maps local displacement, velocity, density error, neighbor offset,
  and phase into residual acceleration.
- **Rationale**: The zero-initialized final layer makes H0 exactly mechanical;
  H1 can measure the benefit of learning without changing the substrate.

### `MechanicalDynamics.step`
- **Does**: Computes Gaussian neighborhood density, excess-density pressure,
  short-range anti-overlap, niche attraction, damping, optional neural residual,
  symplectic integration, and commitment.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| `experiment.py` | One step mutates the reservoir and returns diagnostics | Step signature |
| H1 trainer | Rule features have exactly 12 ordered channels | Feature order/count |
| `metrics.py` | Pressure kernel radius matches reference target density | Kernel semantics |
