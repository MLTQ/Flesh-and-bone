# Kimodo motion review console

## Purpose

The Kimodo review console is a diagnostic tool, not a new flesh-mechanics
experiment. It exists to answer a narrower question before H8 work continues:
does a visibly broken motion originate in Kimodo generation, semantic
retargeting, contact handling, or the later flesh simulation?

Each review job therefore stops at linear-blend skinning. It renders the H4/H5U
character with the generated skeleton but deliberately does not add H7 density
mechanics or learned secondary motion. This isolates the animation and retarget
path from the research system that will eventually consume it.

## Workflow

1. Submit a natural-language prompt, seed, duration, and diffusion-step count to
   the live Kimodo server.
2. Preserve the returned NPZ byte-for-byte.
3. Convert the 77-joint SOMA clip to canonical local rotations and retarget it
   through the existing semantic-role map onto the 24-bone Meshy character.
4. Optionally apply the source heel-contact channels through two-bone IK.
5. Measure anatomical invariants and suspicious pose geometry.
6. Render a camera-following LBS preview from the dense H5U volume and a
   bone-group diagnostic image at the worst pelvis frame.
7. Let the reviewer accept, reject, or annotate the clip. No job is accepted
   automatically.

The full generated sequence is retained. The old H6K half-clip/palindrome
closure is intentionally absent because truncating and reversing a nonperiodic
clip can hide the source motion and introduce an unrelated discontinuity.

## Diagnostic gates

The gates are deliberately broad. They detect gross failures like the observed
76-degree pelvis roll and hip socket above the torso without rejecting athletic
motion merely because it contains high feet or bent joints.

| Measurement | Warning | Failure | Rationale |
| --- | ---: | ---: | --- |
| Maximum pelvis lateral-axis tilt | >20 deg | >35 deg | Known-good samples remain below 8 deg; the broken clip reaches roughly 76 deg. |
| Hip socket above the Hips joint | >20 mm | >50 mm | Known-good samples keep both sockets below Hips; the broken clip raises one by about 117 mm. |
| Parent-child length drift | >0.1 mm | >0.5 mm | Canonical FK should preserve the rest hierarchy exactly. |
| Contact-span foot drift | >20 mm | >50 mm | Contact IK should keep a planted foot near its span-start position. |
| Non-finite values | any | any | A numerical failure invalidates the preview. |

Knee and elbow angles, bilateral hip width, root travel, and the worst frame are
reported for context but are not hard gates. Running, crouching, and stylized
motions can legitimately take those measurements to extremes.

## Artifacts

Jobs are written under `experiments/runs/kimodo_review/<job-id>/`, which remains
ignored until a result is deliberately curated. A complete job contains:

- `raw_kimodo.npz`: exact server response.
- `retargeted_motion.npz`: destination local rotations, root positions, bone
  endpoints, contact channels, and portable metadata.
- `character.gif`: texture-colored, camera-following LBS preview.
- `contact_sheet.png`: evenly sampled overview frames.
- `anatomy_frame.png`: dominant-bone colors and hierarchy at the worst pelvis
  frame.
- `manifest.json`: request, server identity, map coverage, diagnostics, render
  settings, artifact names, and any reviewer decision.

This record is enough to compare several seeds and prompts before deciding
whether the prior failure was a one-off generation, a recurring model failure,
or a retarget convention defect.

## Initial end-to-end validation

The console was validated against the live `kimodo-soma-rp-v1.1` service with
seed 1847 and the prompt “A gentle wave while standing in place, feet planted,
upright posture.” The six-second, 180-frame result maps all 22 destination roles
and passes every gross-anatomy gate:

- raw SOMA FK reconstruction maximum: 0.000000485 m;
- maximum pelvis tilt: 0.428 degrees;
- highest hip socket relative to Hips: -0.0897 m;
- hierarchy length drift: 0.000000221 m; and
- maximum locked-core foot drift after contact IK: 0.000456 m across two spans.

The contact sheet shows a coherent standing wave and the worst-frame bone-group
image keeps the red/blue leg regions on the expected sides. This validates the
console and supplies one good counterexample to the previously malformed clip.
It is not enough to diagnose the prior event as a one-off; several prompts and
seeds should be reviewed through the same manifest before that conclusion.

A second user-submitted sample, seed 948593612 with a relaxed two-step prompt,
is also visually coherent and keeps pelvis tilt below 11.20 degrees, both hip
sockets below Hips, raw FK agreement below 0.4 micrometers, and hierarchy drift
below 0.2 micrometers. The first diagnostic incorrectly reported 51.59 mm of
planted-foot drift because it included the contact solver's intentional
three-frame release blends. The corrected locked-core measurement is 1.27 mm,
so this sample also passes the gross-anatomy screen.

Both previews are intentionally rigid LBS controls. They show whether the
generated skeleton and skin mapping are coherent; they do **not** run H7's local
density, lag, compression, jiggle, or learned secondary flesh mechanics. A
future comparison view should render LBS and frozen flesh mechanics side by side
so animation-source failures and mechanics failures remain distinguishable.
