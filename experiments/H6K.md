# H6K — Kimodo ecological motion transfer

## Question

Do the H6M/H6C conclusions survive a genuinely new skeleton motion rather than
only deterministic retimings of the original walk?

## Motion provenance

Kimodo SOMA-RP-v1.1 generated a 60-frame, 30 Hz clip from:

> a person quickly shifts weight from the left leg to the right leg while
> gently twisting the torso and swinging both arms, staying in place

Seed is `23`; diffusion steps are `50`. Archepelago retargets the raw SOMA
locals onto the existing named 24-bone Meshy rig. The experiment does not use a
generated rig. The first 30 frames are closed as `0..29,28..1`, producing a
58-phase forward/return cycle without duplicate endpoints.

## Pre-evaluation bridge gates

- destination profile is humanoid with confidence at least `0.90`;
- at least 20 canonical roles map;
- root scale lies in `[0.5, 1.5]`;
- identity canonical locals reproduce identity skin matrices within `1e-5`;
- prepared LBS/bone tensors are finite;
- horizontal root travel is at most `0.15 m`.

The prepared artifact already passes these conversion-only gates: confidence
`1.0`, 22 roles, root scale `0.8958`, rest identity error `2.38e-7`, finite
state, and `0.0788 m` horizontal travel. These facts validate the bridge but do
not reveal flesh rollout performance.

## Frozen evaluation

One separately converged graph-elastic teacher uses the exact H5U material and
91,979-cell graph. From its phase-zero state:

1. each of the three frozen H5U MLP checkpoints free-runs for ten cycles, with
   frozen-feature shift and neighbor-blind controls;
2. the H6C constitutive rule is identified only on the original walk, frozen,
   and free-runs on the Kimodo cycle for ten cycles with the same ablation.

No model is trained or normalized on Kimodo states.

## Predeclared acceptance criteria

The Kimodo teacher must pass H6M's finite, seam, bounded-amplitude, and
edge-coherence gates. Every evaluated rule uses the unchanged H6M RMS, p99,
maximum, amplitude, softness, drift, LBS-improvement, finite-state, and
neighbor-causality thresholds.

Results receive three explicit verdicts rather than one blended score:

- **bridge validity:** all conversion gates pass;
- **MLP ecological generalization:** all three H5U checkpoints pass;
- **constitutive ecological generalization:** the original-walk-fitted H6C rule
  passes.

## Interpretation discipline

The palindrome closure makes the clip a valid attractor test but repeats a
forward path backward; it is not evidence for arbitrary nonperiodic motion.
Passing H6C remains expected for a teacher in its exact hypothesis class. The
value of H6K is checking the production retarget/skin path and determining
whether the MLP's measured forcing boundary appears on independently generated
poses and accelerations.

## Results

All three verdicts **pass**.

- The bridge retains humanoid confidence `1.0`, maps 22 roles, uses root scale
  `0.8958`, reproduces identity skin within `2.38e-7`, and limits horizontal
  root travel to `0.0788 m`.
- The external teacher is finite and tightly periodic, with `3.122 mm`
  residual RMS, `2.359` far/near ratio, and `0.311 mm` edge difference RMS.
- Frozen MLP seeds 7/19/31 reach `0.121 / 0.159 / 0.108 mm` RMS and
  `3.112 / 3.498 / 2.856 mm` maximum after ten cycles. Neighbor blindness is
  `2.03–2.76x` worse in position and `6.20–7.16x` worse in edge strain.
- The original-walk-fitted constitutive rule reaches `2.99e-10 m` RMS and
  `5.63e-9 m` maximum.

The Kimodo states are well inside training magnitude: global feature RMS is
`0.381 z`, acceleration p99 is `0.754 z`, and only `0.029%` of values exceed
`3 z`. This explains why genuinely new poses pass while the synthetic fast arm
fails: pose novelty is not the controlling variable; forcing magnitude is.

## Decision

Keep Archepelago/Kimodo as the ecological motion source. The named Meshy rig
does not require the weaker generated-rig path, and the Blender-free canonical
retarget-to-skin bridge is numerically closed. Use generated motion to broaden
pose semantics, but explicitly retime or constrain clips to cover acceleration
ranges; semantic variety alone will not teach high-forcing stability.

Canonical evidence is in `experiments/runs/h6k_final` and the prepared motion
under `experiments/runs/h6k_assets`.
