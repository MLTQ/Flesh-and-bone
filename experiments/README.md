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
