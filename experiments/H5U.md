# H5U — ultra-dense, overlapping flesh splats

## Question

Can H5D scale from 35,527 to roughly 92,000 cells while replacing its sparse,
semi-transparent lattice rendering with overlapping splats that preserve
recognizable texture and facial structure?

## Frozen substrate and appearance

H5U uses a separately stored 12.5 mm volume:

- 91,979 cells (`6.930x` H5 and `2.589x` H5D).
- Grid shape `65 x 137 x 39`, one occupied component, no enclosed pocket.
- Six normalized skin influences and unchanged H4 rig/animation.
- Closest-triangle barycentric UV transfer instead of nearest-vertex color.

The original H4 acceptance intentionally caps its scoped volume at 50,000
cells, so this input reports an H4 failure only on `volume_cell_count`. H5U does
not rewrite that historical gate. It separately requires every H4 provenance,
transport, topology, thickness, weight, and finite-motion gate to pass.

A preregistered render sweep on the corrected 91,979-cell volume freezes:

- base splat radius `0.50 x pitch`;
- per-cell material scale remains `0.75-1.15`;
- maximum world radius `7.1875 mm`, 19.0% above H5D;
- opacity `0.72` instead of `0.52`;
- evidence resolution `720 x 720` instead of `480 x 480`.

The sweep rejected `0.30` because it retained visible rows/holes and `0.60`
because it began smoothing away texture boundaries. Appearance parameters do
not enter mechanics or metrics.

## Density-scaled mechanics

As in H5D, graph coupling scales by inverse pitch squared:

```text
g_ultra = 300 * (0.025 / 0.0125)^2 = 1200
```

Batch size scales with population to the nearest multiple of 256:

```text
round_256(8192 * 91979 / 13273) = 56832
```

The rule remains 17-to-96-to-96-to-3, with 2,400 optimizer steps, four teacher
substeps, 30 warm-up cycles, and three-cycle free rollout. The parameter count
does not grow with cell count.

## Predeclared procedure and acceptance

1. Qualify seed 7 once using only the scaling rules above.
2. If it passes, freeze without tuning and rerun seeds 7, 19, and 31.
3. Retain failures; do not relax inherited H5 thresholds.

H5U inherits every H5 teacher, learning, rollout, stability, LBS, and
neighbor-blind gate. It additionally requires:

- pitch exactly 12.5 mm and at least `6.5x` H5's cells;
- one component, no pocket, normalized weights, finite motion, and barycentric
  UV transfer;
- teacher residual RMS and far/near ratio each within 15% of H5;
- median three-seed rollout RMS at most 0.75 mm and every p99 at most 3.0 mm;
- actual and `4x` learned renders show no systematic see-through grid, torn
  limbs, or loss of the recognizable face/clothing regions seen in the frozen
  static sweep.

## Interpretation discipline

The visual improvement combines more mechanical cells, continuous UV transfer,
greater overlap, opacity, and resolution. It must not be attributed to density
alone. Passing still addresses the same body/walk, not novel motion or biology.

## Results

### Input and visual diagnosis

The first 91,979-cell construction was structurally valid but retained H4's
nearest-vertex UV transfer. Its render repeated only 4,759 vertex colors, so
increasing cell count reduced geometric spacing without adding texture samples.
That result is retained at `experiments/runs/h4_dense_0125`.

The corrected input projects every cell to its closest triangle and
barycentrically interpolates that triangle's corner UVs. The resulting render
recovers the blonde hair, face, white top, belt, green trousers, and boots as
coherent regions. It is retained at
`experiments/runs/h4_dense_0125_bary`. As expected, original H4 acceptance fails
only `volume_cell_count` and aggregate `pass`; every structural gate passes.

The corrected radius/opacity sweep confirms `0.50 / 0.72` as the useful middle
ground. `0.30` remains visibly latticed, while `0.60` begins erasing sharp
clothing and facial boundaries.

### Qualification and teacher continuity

Seed 7 passed on the first predeclared mechanics configuration, so no H5U
hyperparameter tuning was performed. The 523,174-edge teacher has `21.465 mm`
residual RMS (`0.941x` H5), a `2.101` far/near ratio (`0.968x` H5),
`2.220 mm` edge-difference RMS, and a `1.86e-9 m` cycle seam. All inherited
teacher gates pass.

### Final frozen run

| Seed | Holdout accel NRMSE | Rollout RMS | p99 | max | cycle drift | LBS error removed |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.0195 | 0.369 mm | 1.491 mm | 17.737 mm | 0.371 mm | 98.28% |
| 19 | 0.0211 | 0.418 mm | 1.638 mm | 16.649 mm | 0.411 mm | 98.05% |
| 31 | 0.0191 | 0.378 mm | 1.495 mm | 18.593 mm | 0.366 mm | 98.24% |

Every seed passes all H5 gates. Median RMS is `0.378 mm`, 26.3% below H5 and
10.5% below H5D. Learned amplitude is `0.999-1.001` of the teacher and the
far/near ratio remains `2.098-2.101`.

The neighbor-blind control is `4.66-5.31x` worse in position and
`8.16-9.10x` worse in edge strain, strengthening the causal evidence for local
transport at higher density.

The final run took 220.8 seconds on the RTX 4090, including 720 px rendering,
versus 79.1 seconds for H5D's 480 px artifacts. Checkpoints remain about 48 KiB.
Actual and `4x` learned sheets were inspected: the model remains opaque and
recognizable through the cycle with no systematic lattice, tearing, or distant
double structures. Canonical artifacts are in `experiments/runs/h5u_final`.

## Reproduction

```bash
python scripts/run_h4.py \
  --pitch 0.0125 \
  --output experiments/runs/h4_dense_0125_bary

python scripts/run_h5.py \
  --device cuda \
  --volume experiments/runs/h4_dense_0125_bary/volume.npz \
  --steps 2400 \
  --batch-size 56832 \
  --neighbor-coupling 1200 \
  --image-size 720 \
  --render-splat-radius-scale 0.50 \
  --render-opacity 0.72 \
  --output experiments/runs/h5u_final

python scripts/check_h5u.py
```

The H4 command intentionally prints `FAIL` because 91,979 exceeds H4's frozen
50,000-cell scope. `check_h5u.py` verifies that this is the only H4 exception.

## Decision

Keep the 12.5 mm representation, barycentric UV transfer, and `0.50 / 0.72`
rendering. The sparse visual failure was not a mechanics failure: it was the
combination of low overlap, low opacity, low output resolution, and quantized UV
transfer. Further GPU work is deferred; the next mechanics experiment remains
held-out motion generalization rather than another same-walk density increase.
