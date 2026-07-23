# Native runtime lab

## Question

How expensive is the frozen particle NCA at useful body resolutions, and can
physical NCA count be varied independently from the rendered Gaussian count and
radius in a native interactive application?

This is a deployment and instrumentation result, not a new learned-mechanics
experiment. It ports the frozen H6C constitutive backbone and H7C bounded
density residual exactly enough to compare against the Python evaluator.

## Runtime contract

One displayed frame advances:

1. Exact six-influence LBS at previous, current, and next fractional skeleton
   phases.
2. One fixed six-neighbor observation pass and one integration pass per
   substep.
3. Four physics substeps at 120 Hz.
4. One 30 Hz Gaussian-splat render.

The H7C residual is a `5 → 32 → 32 → 2` SiLU MLP with 1,314 learned parameters.
The native model file is 5.3 KB because it also carries two coefficient maxima
and the five H6C coefficients. Model size is effectively irrelevant; body state
and per-cell passes dominate.

| Physical profile | Pitch | Cells | Body asset | Dynamic state |
| --- | ---: | ---: | ---: | ---: |
| coarse | 25.0 mm | 13,273 | 1.6 MB | 2.5 MB |
| dense | 17.5 mm | 35,527 | 4.3 MB | 6.8 MB |
| ultra | 12.5 mm | 91,979 | 11.1 MB | 17.7 MB |

Dynamic state is twelve aligned `float4` arrays per cell: residual and velocity
ping-pong buffers, three LBS samples, two neighbor messages, compression and
stretch vectors, and density scalars. The app also retains canonical points and
sample order on the CPU for view-dependent sorting; this adds about 20 bytes per
cell. Static GPU buffers are approximately the body-asset size.

With the ultra profile active and all three profiles embedded in the bundle,
the complete AppKit/Metal process settled near 132 MiB resident on the
development machine. That number includes framework, shader, window, driver,
CPU sorting, and allocator overhead, so it is intentionally much larger than
the model-plus-state table and is the better desktop deployment estimate.

## Compute measurement

Device: 16-GPU-core Apple M1 Pro. Release build, native Metal, tracked periodic
walk, full H6C + H7C, four substeps per displayed frame, 180 frames. Timings are
the median of five clean, app-closed runs after one warm run. They use Metal
command-buffer GPU intervals and exclude window presentation.

| Cells | Median GPU ms/frame | Five-run range | 30 Hz headroom | RMS residual after 180 | Max residual |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 13,273 | 0.423 | 0.418–0.485 | 78.71× | 18.954 mm | 124.611 mm |
| 35,527 | 1.010 | 1.007–1.016 | 32.99× | 18.445 mm | 128.244 mm |
| 91,979 | 2.491 | 2.481–2.558 | 13.38× | 17.908 mm | 131.146 mm |

All states were finite. The residual magnitudes are the intended visible
secondary motion relative to LBS, not error against a teacher in this runtime
test.

Compute cost is close to linear in cell count. At 92k, it consumes roughly
7.5% of a 33.3 ms 30 Hz frame budget on this laptop GPU. A discrete 4090-class
GPU is unnecessary for inference; it remains useful for training and large
experimental sweeps.

## Numerical parity

The 13,273-cell profile was rolled out for the same 180 frames in PyTorch and
Metal:

| Runtime | RMS residual | Maximum residual |
| --- | ---: | ---: |
| PyTorch float32 | 18.953947 mm | 124.610558 mm |
| Metal float32 | 18.954 mm | 124.611 mm |

An early native arm produced non-finite values because Metal fast-math evaluated
an extreme `tanh` input differently from PyTorch. The final shader clamps that
input to 10 before `tanh`; this is float32-equivalent to saturation and restores
parity. A zero-norm vector branch is also explicit.

## Rendering measurement

The benchmark renders into a 1100×760 BGRA8 target using the production soft
splat shader. It depth-sorts a deterministic uniform sample when the camera,
profile, or render count changes; the sort itself is not in the per-frame GPU
timing. Radius primarily affects fragment coverage, while rendered count affects
vertex and fragment work.

One clean, warmed run produced:

| Physical cells | Rendered cells | 0.75× radius | 1.00× radius | 1.50× radius |
| ---: | ---: | ---: | ---: | ---: |
| 13,273 | 13,273 | 0.111 ms | 0.211 ms | 0.297 ms |
| 35,527 | 35,527 | 0.265 ms | 0.366 ms | 0.567 ms |
| 91,979 | 22,994 | 0.178 ms | 0.169 ms | 0.214 ms |
| 91,979 | 45,989 | 0.335 ms | 0.383 ms | 0.450 ms |
| 91,979 | 91,979 | 0.737 ms | 0.755 ms | 0.961 ms |

The small non-monotonic 25% row is timing noise at sub-0.25 ms duration, not a
claim that larger splats are cheaper. The interactive panel reports the current
configuration continuously.

Canonical rest-pose sorting is a pragmatic translucent-splat solution. It is
visually clean for the current deformation range, but not exact when secondary
motion changes depth order substantially. Weighted blended order-independent
transparency is the likely production replacement.

## Interactive controls and interpretation

- Changing **physical profile** replaces the body graph, resets dynamic state,
  and changes actual NCA compute and memory.
- Changing **rendered cells** changes only visual sample count. A deterministic
  uniform subset avoids chopping off one body region.
- Changing **radius** or **opacity** changes only coverage and fill-rate.
- **H6C + H7C physics** compares living residual motion with pure skinning.
- **H7C density residual** isolates the learned nonlinear correction from the
  structure-preserving H6C backbone.
- **Speed** and **intensity** are broadcast to all cells through the skeleton
  motion source. They are useful interface probes, not an RL policy.

## Orientation and opacity audit

A user inspection at opacity 1.0 initially appeared to show rear hair occluding
the face from the front. Re-rendering the exact production pipeline at named
front, left, back, and right views showed that the reported straight-on image
was the true rear view: the long hair, rear belt pouches, trouser texture, and
foot orientation all match. The adjacent side view correctly reveals the facial
profile. The full-opacity front render shows the face with no rear-hair
occlusion.

This was a UI ambiguity rather than an opacity or depth-sort failure. The lab
now exposes explicit anatomical view buttons while retaining free orbit. The
headless check accepts the same preset and opacity:

```bash
.build/release/FleshAndBoneLab \
  --render-test /tmp/front.png front 1.0
```

## From playback to live intent-driven motion

The useful architectural boundary is:

```text
LLM intent (1–5 Hz)
        ↓
safety / intent arbiter
        ↓
physics-aware skeleton policy (30–120 Hz)
        ↓
joint targets + contacts + root motion
        ↓
rig matrices
        ↓
fixed native flesh NCA (120 Hz)
        ↓
Gaussian renderer
```

The LLM should not emit joint angles or forces directly. It should broadcast a
small, persistent intent such as desired velocity, facing, stance, affect,
gesture, and gaze. A separately trained policy converts that intent plus
proprioception and contact state into stable skeleton motion. Safety constraints
and transition hysteresis sit between the slow semantic source and the fast
controller.

A disciplined path is:

1. Replace the hard-coded walk lookup with a `MotionSource` interface while
   retaining tracked playback as a deterministic control.
2. Add a physics skeleton and a hand-authored PD locomotion controller to prove
   live rig matrices, contacts, and root motion flow through the flesh runtime.
3. Train an RL policy in a fast simulator with command-conditioned locomotion,
   then distill/export only the small inference policy.
4. Add higher-level gesture/pose experts and a hysteretic intent arbiter.
5. Connect an LLM only after the intent schema and safety envelope are stable.
6. Revisit flesh neighborhoods and collision once joint/contact motion exceeds
   the fixed graph's validated deformation range.

The native lab already exercises the last two stages—flesh and rendering—and
provides the performance budget for the controller stages above it.

## Reproduction

```bash
scripts/make_flesh_app.sh
.build/release/FleshAndBoneLab --benchmark 180
.build/release/FleshAndBoneLab --render-benchmark
.build/release/FleshAndBoneLab --render-test /tmp/flesh-native.png
```

The binary body/model assets under `runtime/Assets/`, built app under `dist/`,
and benchmark artifacts under `experiments/runs/` are generated and ignored.
Source experiments, exporter code, runtime code, tests, and this decision record
are tracked.
