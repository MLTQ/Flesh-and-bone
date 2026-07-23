# H6M — frozen-rule motion generalization

## Question

Did H5U learn a reusable local constitutive law, or did it only interpolate the
state manifold of one 29-phase walking loop?

## Frozen evidence and scope

H6M loads the three completed H5U checkpoints (`7`, `19`, and `31`) without
changing their weights, feature normalization, target normalization, graph,
material parameters, or 91,979-cell volume. It constructs a new converged
graph-elastic teacher for each motion and initializes the learned rollout from
that teacher's phase-zero residual and velocity. This privileged initialization
isolates vector-field generalization; it does not test growth, cold-start state
inference, or recovery from an arbitrary residual state.

The original walk is replayed for ten cycles as a calibration arm. It is not
novel-motion evidence.

## Controlled novel motions

All transformations are applied identically to the volume LBS positions and
bone endpoints. They retain the same body, weights, material, graph, 30 Hz
integration clock, and camera.

1. **Reverse:** phase order `0, 28, 27, ..., 1`. The pose set and acceleration
   magnitudes are familiar, but residual velocity and transition direction are
   reversed.
2. **Half speed:** periodic Catmull-Rom resampling from 29 to 58 phases. This
   reduces equilibrium acceleration without duplicating frames.
3. **1.5x speed:** periodic Catmull-Rom resampling from 29 to 19 phases. The
   effective rate is `29 / 19 = 1.526x` and deliberately extrapolates forcing.
4. **Walk then hold:** the 29 walk phases are followed by 15 copies of phase
   zero. This introduces a stop, dwell, and restart while remaining a periodic
   teacher problem.

The resampler is periodic and cubic so the experiment does not manufacture the
piecewise-linear acceleration impulses that ordinary frame interpolation would
create.

## Measurements

For every motion/checkpoint pair, H6M records:

- ten-cycle position RMS, p99, maximum, and phase-zero drift;
- residual-amplitude and far/near softness ratios;
- improvement over the zero-residual LBS control;
- neighbor-blind position and edge-strain degradation;
- finite-state status;
- frozen-normalization feature shift: RMS/p99/max absolute z-score and fraction
  of sampled feature values beyond three training standard deviations.

Feature shift is diagnostic rather than an excuse to renormalize the rule.

## Predeclared acceptance criteria

The original-walk calibration must pass the same gates below for all three
checkpoints over the longer ten-cycle rollout. Each of the four novel motions
must then pass every gate for every checkpoint:

- teacher state finite, cycle seam at most `5e-4 m`, residual RMS between
  `0.001 m` and `0.050 m`, and edge-difference RMS below `0.025 m`;
- rollout RMS at most `0.004 m`, p99 at most `0.012 m`, and maximum at most
  `0.040 m`;
- learned/teacher residual-amplitude ratio in `[0.75, 1.25]`;
- learned far/near amplitude ratio at least `1.25`;
- phase-zero cycle drift at most `0.003 m` and all state finite;
- learned rollout removes at least 35% of LBS-only RMS error;
- neighbor blindness increases position RMS by at least 20% or edge-strain RMS
  by at least 25%.

The primary H6M verdict is deliberately strict: all three seeds times all four
novel motions must pass. A partial result will be reported by motion and seed,
not converted into a pass by relaxing thresholds after the run.

## Kimodo ecological arm

After the controlled verdict, one Kimodo clip may be added as an exploratory
ecological test. It cannot change the controlled H6M pass/fail result. The clip
must use the existing named 24-bone rig—no new generated rig—and its retarget
report must record mapped roles, root scale, finite matrices, and a rest-pose
identity check before flesh metrics are interpreted. This prevents rigger or
coordinate-conversion error from masquerading as failed mechanics.

## Interpretation discipline

Passing demonstrates frozen-rule generalization across temporal forcing on one
body and synthetic material law. It is not evidence for cross-body,
large-deformation, collision, incompressibility, or biological realism.
Failure localizes the next curriculum: direction failure suggests missing
state coverage, fast/stop-only failure suggests forcing extrapolation, and
normalization-tail failure suggests the rule needs a dimensionless physical
parameterization rather than merely more samples of one walk.

## Results

H6M completes in 232 seconds on the RTX 4090 and **fails** its strict aggregate
criterion. The failure is sharply structured rather than a uniform loss of
motion capability.

| Motion | Seed 7 RMS / max | Seed 19 RMS / max | Seed 31 RMS / max | Verdict |
|---|---:|---:|---:|---|
| replay | 0.377 / 17.7 mm | 0.431 / 27.0 mm | 2.838 / 167.9 mm | fail |
| reverse | 0.742 / 45.6 mm | 0.832 / 48.8 mm | 1.617 / 182.4 mm | fail |
| half speed | 0.165 / 4.33 mm | 0.197 / 4.32 mm | 0.167 / 4.50 mm | **pass** |
| `1.526x` | 5,474,184 / 1,166,177,875 mm | 52.307 / 10,596.9 mm | 19.143 / 338.8 mm | fail |
| walk then hold | 0.774 / 34.6 mm | 0.675 / 26.7 mm | 1.196 / 171.0 mm | fail |

Replay seeds 7 and 19 remain stable over the longer horizon. Seed 31 develops a
rare but severe tail outlier, establishing that the old three-cycle pass did
not guarantee ten-cycle stability. Reverse retains low RMS and correct global
amplitude for every seed but misses the unchanged 40 mm worst-cell gate. Seeds
7 and 19 pass stop/dwell; seed 31 again fails only the maximum tail gate.

The clean half-speed pass and catastrophic fast-motion failure align with
frozen feature support. Half speed has acceleration p99 `1.11 z` and only
`0.049%` of sampled feature values beyond `3 z`; fast motion reaches
acceleration p99 `7.62 z` with `7.70%` of all values beyond `3 z`. Reverse is
near the training distribution and fails through sparse tails rather than bulk
error.

## Decision

Do not solve this by merely adding more hidden channels or relaxing maximum
error. The local state is sufficient for slow, reversed, stop/dwell, and later
Kimodo motion, but the unconstrained MLP has no stable extrapolation law and one
checkpoint is marginal even on long replay. Run a structure-preserving
constitutive control, then retain the MLP only as a bounded residual around a
stable backbone.

Canonical evidence is in `experiments/runs/h6m_final`.
