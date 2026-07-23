# Experiment ledger

This file is the chronological decision record for Flesh and Bone. Each run gets
its own directory containing the resolved configuration, metrics JSON, visual
artifacts, and a short Markdown verdict. Checkpoints and generated media are not
accepted as evidence without the matching configuration and metrics.

## H0 — five-bone mechanical control

**Question:** Can a fixed-capacity population of mobile Gaussian cells be fed
into a five-bone H body plan, maintain regulated density, differentiate into a
persistent checker material, preserve the two negative spaces, and follow an
animated skeleton?

**Control:** Explicit unfilled-niche reservation, damped attraction to the
moving niche, and local over-density pressure. No learned residual.

**Planned acceptance criteria:**

- Final target coverage at least 0.90.
- Mean committed tracking error no more than 0.10 world units.
- At least 0.95 of target cells committed.
- Crowding fraction below 0.08.
- No more than 0.02 of active mass in either negative-space probe.
- Moving-phase coverage loss no greater than 0.08.
- No non-finite state and no unaccounted mass.

**Status:** complete; mechanical control passes on seeds 7, 19, and 31. See
[`H0.md`](H0.md) for the evidence and interpretation.

**Decision:** keep the particle/splat substrate. H0 establishes geometry,
rendering, feeding, persistent differentiation, motion tracking, metrics, and
artifact contracts. It does **not** establish emergent density-driven assembly:
one-to-one niche reservation makes the final equilibrium intentionally easy and
leaves pressure almost inactive. The next experiment must remove that privilege.

## H1 — continuous deficit recruitment

**Question:** Can uncommitted particles discover unfilled tissue from a smooth
bone-conditioned deficit field, without receiving a unique target-site index?

**Planned controls:**

- H0 unique-niche upper bound.
- Continuous deficit with density pressure.
- Continuous deficit with pressure disabled.
- Nearest-bone-only attraction, expected to overfill junctions/erase gaps.

**Primary evidence:** allocation entropy, gap occupancy, coverage, density
relative error, transient crowding, differentiation correctness, and assignment
leakage. Material evidence separately requires at least 0.80 checker-field
accuracy over covered sites and a fully locked material state during motion; a
global navy/cream histogram is not sufficient. Damage/re-feeding remains a
follow-up because H1 first had to establish stable unassigned assembly.

**Status:** complete as a first continuous-field baseline. The main arm passes
all gates on two of three seeds; the third misses only the strict moving
coverage gate. Pressure-off collapses, proving pressure is causally required.
Nearest-bone plus pressure also passes, showing that the uniform capsule H is
not sufficiently discriminating to prove live deficit sensing is required. A
plastic-to-locked local material field repairs the collapsed H1 checker texture
on all three seeds; the first-contact control fails the spatial metric. See
[`H1.md`](H1.md).

## H2 — nonuniform morphogenesis and wound repair

**Question:** Can freely fed cells allocate across visibly nonuniform anatomy,
then can a second generation selectively repair an erased region and retain its
material identity during motion?

**Primary arm:** Live regional-capacity allocation assigns each uncommitted
cell one persistent coarse anatomical guide. Within that region, continuous
deficit attraction and density pressure determine position; no target-site
identity or final coordinate is exposed. New cells lock only after consecutive
local stability observations.

**Status:** complete. The frozen primary configuration passes every predeclared
gate on seeds 7, 19, and 31. A nearest-bone/uniform-density arm cannot allocate
the nonuniform anatomy or repair the remote wound; pressure-off collapses; and
first-contact material preserves geometry but loses the checker chart.

**Decision:** keep the sparse particle representation and the two-scale
developmental decomposition. H2 establishes that a coarse fate plus continuous
local placement is sufficient for this repair task and is much less privileged
than one-site-per-cell reservation. It does not establish autonomous local
morphogenesis or a learned NCA: the seven-way guide reads global regional
capacity, and the neural residual is zero. This motivates learned fate and
residual-motion experiments against the controller rather than silently
treating the guide as emergent. H3 completes the fate-distillation half;
learned local motion remains open. See [`H2.md`](H2.md).

## H3 — fine humanoid envelope and learned fate

**Question:** Can the substrate support a fine, variable-thickness humanoid
whose cheeks and buttocks are not bone offsets, and can a learned fate scorer
replace H2's hand-written regional decision during both growth and remote
repair?

**Primary arm:** A shared per-region MLP is trained on randomized states from
the frozen capacity/distance teacher, then frozen before rollout. It chooses a
persistent coarse fate from shortage, availability, and distance. Continuous
within-region deficit and pressure remain mechanical.

**Status:** complete. The 15-bone, 18-region, 1,092-cell representation passes
all declared geometry gates. Oracle and learned arms both pass every applicable
gate on seeds 7, 19, and 31. The learned model scores 0.984–0.997 held-out
agreement, moving coverage remains 0.855–0.878, and the remote right-cheek wound
recovers to 0.813–0.875. Local-only and shortage-blind controls never assemble
the cheek and fail globally.

**Decision:** retain the humanoid particle substrate and global-fate/local-place
decomposition. H3 establishes learned fate scoring and meaningful
extra-skeletal flesh at finer resolution. It does not establish end-to-end NCA
control: learning is supervised oracle distillation, regional demand is global,
and acceleration remains mechanical. H4 should import a production human rig
and learn local flesh dynamics against this measured upper bound. See
[`H3.md`](H3.md).

## H4 — production rig and variable-thickness human target

**Question:** Can a real skinned FBX be converted into a runtime-independent
rig, numerically reproduce Blender's walk, and seed a fine watertight human
volume whose mass is not a bone offset?

**Arms:** Full skin weights establish import correctness. Normalized top-3,
top-4, and top-6 weights determine the smallest safe persistent state. Surface
and filled-volume arms then run the same 30-frame loop.

**Status:** complete; every predeclared gate passes. Full skinning matches
Blender at `1.10e-7 m` RMS. Top-3 and top-4 fail; top-6 passes at 0.106 mm RMS
and 0.498 mm p99. The filled target contains 13,273 connected 2.5 cm cells, has
no enclosed pocket, places 62.0% of mass more than 8 cm from any bone, and uses
at most 8.625 mm rendered splats.

**Decision:** keep the imported 24-bone rig, six-influence state, exact surface
oracle, and variable-thickness volume. H4 solves asset transport and target
construction, not flesh mechanics. H5 should generate a soft-tissue teacher
driven by the imported bones and learn local residual forces relative to the
measured LBS baseline. See [`H4.md`](H4.md).

## H5 — separately taught local flesh mechanics

**Question:** Can one shared local particle-message rule learn stable secondary
motion from an explicit graph-elastic teacher instead of incorrectly treating
the imported LBS walk as flesh ground truth?

**Arms:** A warmed periodic graph-elastic teacher provides the curriculum. The
learned arm free-runs autoregressively for three cycles. LBS-only supplies the
zero-residual baseline, while a neighbor-blind rollout zeros only the two local
message vectors in the same checkpoint.

**Trial history:** The first frozen 1,200-step run failed honestly: seed 31 had
a 44.71 mm worst-cell error against the 40 mm gate, localized to one rare
degree-2 soft-surface cell. Seeds 7 and 19 otherwise passed. The threshold and
teacher were not changed. Doubling exposure to 2,400 steps reduced seed 7's
permitted tuning maximum from 30.32 to 16.81 mm, after which the configuration
was frozen and all seeds rerun.

**Status:** complete. Seeds 7, 19, and 31 all pass. Three-cycle learned error is
0.478-0.528 mm RMS, 1.879-2.178 mm p99, and 16.051-19.619 mm maximum. Cycle
drift remains below 0.500 mm and the learned rule removes 97.69-97.90% of the
LBS-only error. The neighbor-blind control is 3.88-4.19x worse in position and
5.31-5.89x worse in edge strain.

**Decision:** retain the local message architecture and its explicit-teacher
workflow. This demonstrates stable distillation and causal use of local
transport on one body/walk, not biological accuracy or generalization. The
next mechanics test should freeze the rule under novel accelerations before
introducing density, contact, incompressibility, or muscle curricula. See
[`H5.md`](H5.md).

## H5D — high-density local flesh scaling

**Question:** Does H5 remain stable when the same body is discretized with
materially more and smaller cells?

**Protocol:** Pitch falls from 25 mm to 17.5 mm, producing 35,527 cells
(`2.677x` H5). Graph coupling scales by inverse pitch squared, from 300 to
612.2449, to preserve continuum stiffness. Batch size scales from 8,192 to
22,016 while optimizer steps and all H5 gates remain fixed. Seed 7 qualifies
the predeclared scaling once; seeds 7, 19, and 31 then rerun from scratch.

**Status:** complete. Seed 7 passes without tuning and all final seeds pass.
Rollout RMS is 0.406-0.468 mm, p99 is 1.612-1.827 mm, maximum is
13.810-20.278 mm, and cycle drift stays below 0.431 mm. Median RMS improves
17.6% from H5. The neighbor-blind control is 4.29-4.94x worse in position and
6.85-7.51x worse in edge strain.

**Decision:** retain the denser discretization and pitch-scaled coupling. The
same 48 KiB shared rule scales to 2.68x the cells in 2.08x the wall time. A
further same-motion density sweep is now less informative than frozen-rule
tests on unseen accelerations, followed by density/contact mechanics. See
[`H5D.md`](H5D.md).

## H5U — ultra-dense overlapping splats

**Question:** Can the system reach roughly 92k cells and remove the visible
grid/see-through artifact without sacrificing learned dynamics or blurring away
recognizable surface detail?

**Visual diagnosis:** More cells alone repeated nearest-vertex UV colors and
could not create facial or clothing detail. H5U switches to closest-triangle
barycentric UVs, freezes a `0.50 x pitch` radius after an explicit sweep,
raises opacity to 0.72, and renders at 720 px. The corrected character is
opaque and recognizable across the walk.

**Protocol:** The 12.5 mm volume contains 91,979 cells (`6.930x` H5). Coupling
scales to 1200 and batch size to 56,832; model size and optimizer steps remain
fixed. The historical H4 gate fails only its intentional 50k resource ceiling.
Seed 7 qualifies without tuning, then all seeds rerun from scratch.

**Status:** complete. Seeds 7, 19, and 31 pass. Rollout RMS is
0.369-0.418 mm, p99 is 1.491-1.638 mm, maximum is 16.649-18.593 mm, and median
RMS improves 26.3% over H5. Neighbor blindness is 4.66-5.31x worse in position
and 8.16-9.10x worse in edge strain.

**Decision:** keep the ultra-dense volume, barycentric UV transfer, and
overlapping render. The former sparse appearance was a representation/render
transfer problem, not failed mechanics. Pause GPU scaling here; test the frozen
rule on unseen motion next. See [`H5U.md`](H5U.md).

## H6M — frozen-rule motion generalization

**Question:** Do the three frozen H5U rules implement a reusable local flesh
law, or only the state manifold of the original 29-phase walk?

**Protocol:** Replay calibrates the evaluator over ten cycles. Four predeclared
novel forcing cycles—reverse, periodic half speed, periodic `1.526x` speed, and
walk-then-hold—then run without changing checkpoint weights or normalization.
Every motion gets a separately converged graph-elastic teacher, a neighbor-blind
control, long-horizon rollout metrics, and frozen-feature z-shift diagnostics.

**Status:** complete; strict aggregate **fail**. Half speed passes every seed,
reverse remains low-RMS but misses rare-cell maximum gates, stop/dwell passes
seeds 7 and 19, and `1.526x` destabilizes every MLP. Seed 31 also exposes a
ten-cycle replay tail failure. Feature tails track forcing magnitude. See
[`H6M.md`](H6M.md).

## H6C — structure-preserving constitutive identification

**Question:** Is H6M's fast-motion instability caused by missing local state or
by an unconstrained MLP extrapolating a stable physical law?

**Protocol:** Identify five shared coefficients on the same original-walk
teacher states, freeze them, then rerun the exact H6M ten-cycle motion and
neighbor-ablation suite. This is an explicit structural upper-bound control,
not a claim that the synthetic teacher is real tissue.

**Status:** complete; **pass**. The five coefficients recover the teacher to
`5.48e-8` held-out NRMSE and every ten-cycle motion stays below `9.5e-8 m`
maximum, including the failed fast arm. This localizes H6M's failure to
unconstrained regression. See [`H6C.md`](H6C.md).

## H6K — Kimodo ecological motion transfer

**Question:** Do the temporal-forcing results survive a newly generated and
retargeted skeleton motion?

**Protocol:** Kimodo generates an in-place weight-shift/torso-twist clip;
Archepelago retargets it onto the existing named Meshy rig; a 30-frame segment
is palindrome-closed to 58 phases. Frozen H5U MLPs and the original-walk-fitted
H6C rule then run the unchanged ten-cycle protocol with separate verdicts.

**Status:** complete; all three bridge/MLP/constitutive verdicts **pass**. Frozen
MLPs reach `0.108–0.159 mm` RMS and under `3.5 mm` maximum. Kimodo feature
magnitudes are far below the failed fast arm, showing that new pose semantics
generalize when acceleration remains supported. See [`H6K.md`](H6K.md).

## H7 — bounded hybrid density mechanics

**Question:** Can the frozen H6C constitutive backbone acquire local nonlinear
compression pressure and stretch cohesion through a bounded invariant residual,
while remaining stable on severe and externally generated motion?

**Protocol:** A synthetic density teacher adds clipped LBS-relative edge-strain
forces capped at `12 m/s²`. A small MLP predicts only two bounded nonnegative
coefficients multiplying explicit equivariant vectors. Replay, reverse, and
walk-then-hold train the residual; half speed seed 7 qualifies it; fast and
Kimodo motions remain untouched final holdouts. See [`H7.md`](H7.md).

**Status:** initial qualification **failed as vacuous**. The hybrid learned the
new force accurately and remained stable, but teacher density RMS and backbone
error fell below their predeclared nontriviality floors. Final holdouts stayed
sealed. See [`H7.md`](H7.md).

## H7B — non-vacuous bounded density mechanics

**Question:** Does the same bounded architecture still learn and remain stable
when the nonlinear density term is large enough to matter?

**Protocol:** Scale only pressure/cohesion and their learned maxima by 8x, based
on H7's measured half-speed shortfall. Keep the strain definition, 12 m/s² cap,
architecture, optimizer, split, and every gate unchanged. See
[`H7B.md`](H7B.md).

**Status:** qualification **failed one non-vacuity gate**. Density force became
nontrivial and every other gate passed, but backbone error was `0.1438 mm`
against the unchanged `0.2 mm` floor. Final holdouts stayed sealed. See
[`H7B.md`](H7B.md).

## H7C — capped non-vacuous density mechanics

**Question:** Can a 24x-original density scale clear H7's fixed non-vacuity
floor while the unchanged smooth acceleration cap preserves stability?

**Protocol:** Scale only the teacher coefficients and learned maxima to 3x H7B.
All features, dead zones, architecture, optimizer, splits, gates, and the 12
m/s² cap remain unchanged. See [`H7C.md`](H7C.md).

**Status:** strict aggregate **fail**. Qualification and the severe fast holdout
pass all seeds, with fast error falling from `4.010 mm` backbone RMS to
`0.117–0.119 mm`. Untouched Kimodo is stable and nearly exact but fails only the
fixed non-vacuity gate: its backbone error is `0.0969 mm`. See
[`H7C.md`](H7C.md).

## H7D — frozen Kimodo excitation audit

**Question:** Is H7C's Kimodo failure a missing generalization capability or
simply insufficient nonlinear excitation?

**Protocol:** Keep H7C checkpoints frozen. Treat untouched Kimodo as a stability
arm and deterministically resample the same clip from 58 to 29 phases for a 2x
temporal stress that must pass every original H7 gate. See [`H7D.md`](H7D.md).

**Status:** complete; **pass**. Untouched Kimodo passes ecological stability. At
2x speed the backbone error becomes a non-vacuous `0.477 mm`, while all frozen
hybrids remain near `0.001 mm` and pass every original H7 gate. See
[`H7D.md`](H7D.md).

## Interpretation discipline

- H0 validates mechanics and instrumentation, not neural self-organization.
- A hard niche assignment is an upper-bound control. H1–H3 progressively
  remove that privileged information.
- H1 phenotype may remain plastic during stationary assembly, but must lock
  before motion. Recomputing it from world coordinates after lock would hide
  material sliding.
- Filling either central gap is a failure even if global coverage improves.
- Added particles must follow tissue deficit; nearest-bone distance alone is not
  considered a sufficient general developmental signal, even though the H1
  control shows it is sufficient for this uniform tube morphology.
- A coarse anatomical fate is weaker than a target site but remains privileged
  information. Results using it must report guide exposure separately and must
  not be described as purely local self-organization.
- A learned fate score is not an end-to-end learned organism when its labels and
  inputs expose an oracle-defined global region-demand vector. H3 is reported as
  distillation and tested with a shortage-blind causal control.
- A linearly skinned animation validates rig transport but contains no target
  elasticity or secondary motion. H4 must not be cited as learned flesh.
- A graph-elastic teacher is a synthetic curriculum, not measured human tissue.
  H5 may be cited as stable local-rule distillation on one body and walk, not as
  biological realism or cross-motion generalization.
- H5D establishes discretization scaling on that same body and walk. More cells
  do not convert a same-trajectory distillation result into generalization.
- H5U's appearance improvement combines density, barycentric texture transfer,
  overlap, opacity, and resolution; it must not be attributed to density alone.
- H6M grants each frozen rule the novel teacher's converged phase-zero state.
  Its verdict concerns vector-field generalization, not cold-start inference.
- H6C's hypothesis class contains the graph-elastic teacher exactly. A pass
  diagnoses H6M's regression parameterization; it is not independent physics.
- H6K's forward/return Kimodo cycle validates an external pose/retarget path,
  not arbitrary nonperiodic animation.
