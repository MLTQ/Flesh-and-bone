# `density_teacher.py`

## Intent

This module introduces H7's first deliberately nonlinear flesh mechanism while
retaining H6C's stable graph-elastic backbone. It measures local axial strain in
the moving LBS frame, converts excess compression and stretch into explicit
equivariant vectors, and supplies a bounded synthetic training target.

## Contracts

- The graph is the directed six-neighbor voxel graph from `flesh_teacher.py`.
- Equilibrium edge directions are recomputed from the current LBS phase; the
  nonlinear signal is therefore residual deformation, not skeleton motion.
- Five-percent compression and eight-percent stretch are dead zones frozen by
  the pre-H7 strain diagnostic.
- Squared excess strain makes onset continuous and weights severe local events.
- Pressure points away from compressed neighbors; cohesion points toward
  stretched neighbors. Directed reverse edges make internal pairs balanced up
  to boundary and degree averaging effects.
- `smooth_norm_cap` preserves direction and asymptotically bounds every cell's
  density acceleration at 12 m/s².
- Captured scalar/vector observations are supervision for the learned bounded
  residual. The full acceleration remains the sum of the linear backbone and
  the density term.

## Deliberate limitations

The equilibrium graph is fixed and derived from a voxelized body. There is no
collision detection, changing neighbor topology, incompressibility constraint,
plasticity, muscle drive, or measured material data. LBS itself supplies the
moving reference configuration. These restrictions keep H7 a sharp test of
safe nonlinear extension rather than an overclaimed flesh simulator.
