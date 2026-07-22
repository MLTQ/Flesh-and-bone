# H5D — high-density local flesh scaling

## Question

Does H5's learned local flesh mechanism survive a materially denser particle
discretization, rather than succeeding only because the 25 mm grid is small?

## Frozen substrate

H5D reduces H4 pitch from 25 mm to 17.5 mm. The resulting volume is generated
before learning and stored separately; it must not overwrite the H4/H5
baseline.

- Baseline: 13,273 cells at 25 mm pitch.
- H5D: 35,527 cells at 17.5 mm pitch (`2.677x` as many cells).
- Dense grid: `47 x 98 x 29`, one occupied component, no enclosed pocket.
- Maximum rendered splat radius: 6.0375 mm versus 8.625 mm at baseline.
- Six normalized skin influences and the original rig/animation remain fixed.

The dense H4 construction has already passed the unchanged H4 geometry gates;
this is input validation, not a learned result.

## Density-scaled mechanics

A raw graph Laplacian shrinks approximately with pitch squared for the same
smooth physical deformation. To preserve H5's continuum-scale neighbor
stiffness, H5D freezes:

```text
g_dense = 300 * (0.025 / 0.0175)^2 = 612.2448979591837
```

All other teacher parameters, 30 warm-up cycles, four substeps, model width,
features, three-cycle rollout, and acceptance thresholds remain unchanged.

Training exposure scales with cell count. H5 used batches of 8,192; H5D uses
22,016 (the nearest multiple of 256 to `8192 * 35527 / 13273`) for the same
2,400 optimizer steps. This preserves approximately equal sampled examples per
cell while exploiting the 4090's available memory.

## Predeclared procedure

1. Run seed 7 as a qualification arm using only the scaling rules above.
2. If seed 7 passes, freeze the configuration without further tuning.
3. Run fresh seeds 7, 19, and 31 and require all three to pass.
4. Retain any failed trial and never relax an H5 threshold after observing it.
5. Compare actual-scale and explicitly labeled `4x` residual renders with H5.

## Predeclared acceptance

H5D inherits every H5 teacher, one-step, rollout, stability, LBS, and
neighbor-blind gate. In addition:

- Cell count is at least `2.5x` H5 and pitch is exactly 17.5 mm.
- The dense volume has one occupied component, no enclosed pocket, normalized
  weights, finite motion, and maximum splat radius at most 6.1 mm.
- Teacher residual RMS remains within 15% of H5's `22.813 mm` and teacher
  far/near amplitude ratio remains within 15% of H5's `2.169`.
- Across the final three seeds, median rollout RMS is at most 0.75 mm and every
  p99 is at most 3.0 mm. The inherited 40 mm maximum gate remains authoritative.

## Interpretation discipline

Passing H5D establishes discretization scaling on the same body and walk. It
still does not establish novel-motion, novel-body, contact, incompressibility,
muscle, growth, or damage generalization.

## Results

### Qualification

Seed 7 passed on the first predeclared density-scaled configuration, so no H5D
hyperparameter tuning was performed. Its rollout RMS was `0.424 mm`, p99
`1.690 mm`, and maximum `13.529 mm`.

### Teacher continuity

The dense graph has 198,310 directed edges and one component. Its teacher
residual RMS is `22.120 mm`, only 3.0% below H5, and its `2.121` far/near ratio
is only 2.2% below H5. The `1.83e-9 m` cycle seam and `3.216 mm` edge-difference
RMS pass the inherited gates. Scaling coupling by inverse pitch squared
therefore preserved the intended physical regime across discretizations.

### Final frozen run

| Seed | Holdout accel NRMSE | Rollout RMS | p99 | max | cycle drift | LBS error removed |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.0213 | 0.423 mm | 1.683 mm | 13.810 mm | 0.417 mm | 98.09% |
| 19 | 0.0219 | 0.468 mm | 1.827 mm | 20.278 mm | 0.430 mm | 97.88% |
| 31 | 0.0195 | 0.406 mm | 1.612 mm | 15.175 mm | 0.387 mm | 98.16% |

All inherited H5 gates and all H5D-specific gates pass on every seed. Median
rollout RMS is `0.423 mm`, 17.6% lower than H5's `0.513 mm`; every p99 remains
below `1.83 mm`. Learned amplitude is `0.998-1.000` of the teacher and the
far/near ratio remains `2.115-2.124`.

The neighbor-blind control is `4.29-4.94x` worse in position and
`6.85-7.51x` worse in edge strain. Local transport becomes more, not less,
diagnostically important at higher density.

The frozen three-seed run took 79.1 seconds on the RTX 4090 versus 38.1 seconds
for H5: `2.08x` wall time for `2.68x` as many cells. Checkpoints remain about
48 KiB because the shared rule's parameter count is independent of population
size. Actual and labeled `4x` residual sheets were visually inspected and show
the same coherent motion with substantially finer surface sampling.

Canonical artifacts are in `experiments/runs/h5d_final`; the validated dense
input construction is in `experiments/runs/h4_dense_0175`.

## Reproduction

```bash
python scripts/run_h4.py \
  --pitch 0.0175 \
  --output experiments/runs/h4_dense_0175

python scripts/run_h5.py \
  --device cuda \
  --volume experiments/runs/h4_dense_0175/volume.npz \
  --steps 2400 \
  --batch-size 22016 \
  --neighbor-coupling 612.2448979591837 \
  --output experiments/runs/h5d_final

python scripts/check_h5d.py
```

## Decision

Keep the denser representation and density-scaled local rule. H5D rejects the
hypothesis that H5 only worked at low particle count. The most informative next
step is no longer another same-motion density increase: freeze the mechanics
and test unseen motion spectra, followed by explicit density/contact curricula.
