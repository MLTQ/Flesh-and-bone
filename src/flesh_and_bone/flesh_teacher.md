# flesh_teacher.py

## Purpose

Defines H5's explicit sparse graph-elastic teacher over the H4 volume. It adds
small secondary motion around LBS without treating the imported linear-skinned
mesh as flesh-mechanics ground truth.

## Components

### `ElasticTeacherConfig`
- **Does**: Records phase rate/substeps/warmup plus bone-distance stiffness,
  damping ratio, and neighbor coupling.

### `VoxelGraph` / `build_voxel_graph`
- **Does**: Reconstructs directed six-neighbor edges from regular volume points
  and independently counts connected components.

### `neighbor_mean_difference`
- **Does**: Computes a sparse local graph Laplacian by indexed accumulation.

### `volume_lbs_cycle`
- **Does**: Skins the 29 unique phases and computes periodic central-difference
  equilibrium acceleration.

### `periodic_acceleration` / `simulate_teacher_from_lbs`
- **Does**: Computes closed-cycle central-difference forcing and converges the
  same graph-elastic material around any uniformly sampled periodic LBS
  trajectory.
- **Rationale**: H6M must change skeletal forcing without changing material
  mechanics or manufacturing a new training path.

### `simulate_teacher`
- **Does**: Skins the H4 walk, then delegates to the arbitrary-cycle teacher.

## Contracts

| Dependent | Expects | Breaking changes |
|---|---|---|
| H5 learner | Captured tensors shaped `(phase, substep, cell, 3)` | Layout/order |
| H5 metrics | Final state follows exactly one captured cycle | Capture timing |
| H4 asset | Duplicate last animation endpoint is excluded | Phase convention |
| H6M experiment | Arbitrary cycles use `config.fps` and identical mechanics | Time/physics drift |

## Notes

The teacher is an under-damped small-strain curriculum. It intentionally omits
contact, incompressibility constraints, plasticity, and biological calibration.
