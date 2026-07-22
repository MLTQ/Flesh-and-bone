# Flesh and Bone

Flesh and Bone is an experimental particle neural cellular automaton for
skeletally coherent, self-assembling creatures. Instead of treating every voxel
in a dense volume as a stationary NCA site, it represents living material as a
sparse reservoir of persistent mobile Gaussian splats. Splats can be introduced
as undifferentiated cells, migrate toward under-filled tissue niches, acquire a
body-part identity, and then move with an articulated skeleton while maintaining
local density.

The project asks whether a useful creature can be decomposed into:

- **Bone**: a low-dimensional articulated scaffold providing global pose,
  velocity, and bone-local coordinate frames.
- **Body plan**: bone-conditioned morphogen fields describing desired tissue,
  density, and surface organization without dictating particle trajectories.
- **Flesh**: persistent Gaussian-splat agents carrying position, velocity, mass,
  appearance, differentiation, attachment, and recurrent neural state.
- **Guide**: a global hard state/transition controller, eventually using the
  attractor-and-directed-edge mixture-of-experts pattern validated in Bonsai.

This is not conventional linear-blend skinning. The skeleton supplies a coarse
motion scaffold; local particle rules remain responsible for pressure,
cohesion, repair, migration, differentiation, secondary motion, and appearance.

## Why particles rather than a dense NCA volume?

A dense NCA grid is an Eulerian substrate: space is stationary and creature
state flows between sites. That is efficient and easy to train, but individual
cells have no persistent physical identity and most of a large 3D volume may be
empty. A particle NCA is Lagrangian: the agents themselves move. This makes
lineage, feeding, mass conservation, differentiation, and variable-resolution
flesh natural concepts.

The tradeoff is significant. Particle systems require stable neighborhood
search, target-density homeostasis, collision handling, and controlled
birth/death. Gaussian splats also need *some* overlap to make a continuous
surface, so the desired mechanic is not universal repulsion. It is tissue
homeostasis: cohesion below a target density and pressure above it.

The likely long-term design is particle-in-cell. A sparse splat population
carries identity and appearance, while a coarse grid measures density, pressure,
tissue demand, and collision fields. The first experiment uses direct particle
neighborhoods so this hypothesis can be tested with minimal infrastructure.

## Experiment H0: five-bone H scaffold

The initial body plan is a planar five-bone **H** embedded in 3D:

```text
top-left             top-right
    |                     |
    |                     |
 left-junction -------- right-junction
    |                     |
    |                     |
bottom-left          bottom-right
```

Each central junction has degree three—a double-ended Y. The four vertical
segments plus the bridge are the five bones. Capsule-shaped tissue around the
bones leaves two deliberately empty regions above and below the bridge, between
the uprights. Those negative spaces are useful: a rule that merely fills the
convex hull will visibly fail.

The reference skin uses a canonical X/Y checker chart extruded through tissue
thickness. Checker identity is attached to material coordinates, not recomputed
from current world position during motion, so coherent animation is easy to
inspect. A volumetric XYZ parity lattice was rejected because overlapping depth
layers collapse into visual speckles rather than a readable skin. The skeleton
first remains still while cells are fed, then performs a bounded jiggle and spin.

### H0 state

Each reservoir entry stores:

- Position and velocity.
- Active mass and commitment state.
- Assigned developmental niche.
- Dominant body part and soft bone weights.
- Persistent checker material identity.
- Gaussian radius/opacity for rendering.
- Reserved recurrent features for the learned particle NCA.

The explicit body-plan control assigns one cell to each unfilled niche, attracts
it toward that moving niche, and adds density-dependent pressure when neighbors
overcrowd. This is intentionally a control, not the research result. Later
experiments progressively remove its privileged information before using it as
a teacher for a learned message-passing residual.

### H0 measurements

The baseline reports:

- Target-site coverage and mean tracking error.
- Density error and short-range crowding.
- Commitment/differentiation rate.
- Occupancy of the two intended negative spaces.
- Motion tracking after the skeleton begins moving.
- Active mass and reservoir accounting.

Every run writes a JSON metrics record, a PNG contact sheet, an animated GIF,
and a Markdown note under `experiments/`. Results are evidence; a visually
pleasing animation without the negative-space and density checks is not a pass.

**Current status:** H0 passes on three seeds. It validates the substrate and
instrumentation but remains a privileged one-niche-per-cell upper bound. H1
removes unique destinations and tests continuous tissue-deficit recruitment.
See [`experiments/H0.md`](experiments/H0.md).

## Experiment H1: continuous tissue-deficit recruitment

H1 removes the per-cell developmental niche. Cells enter with
`assigned_site = -1`, observe a smooth raster of current tissue deficit, and
move under local deficit attraction plus density pressure. Contact begins a
plastic material phase: cells read local checker phenotype while their soft
bone coordinates continue to settle. Immediately before motion the phenotype
and attachment are locked and remain persistent; no target-sample identity is
stored. The rendered Gaussian radius is `0.20 × spacing`, down from H0's
`0.31 × spacing`, so cell-scale structure and crowding are easier to see.

The main arm passes all gates on seeds 7 and 31; seed 19 misses only the strict
moving-coverage gate at 0.883 while retaining correct density, tracking,
commitment, and negative space. The pressure-off arm collapses, establishing
that density pressure is causal. A nearest-bone-plus-pressure control also
passes on this uniform capsule H, so this morphology does not yet establish
that live deficit sensing is necessary. Spatial checker accuracy after motion is
0.923, 0.928, and 0.915 across the three seeds; the legacy first-contact rule
scores 0.561 on seed 7 despite retaining a plausible global color balance. See
[`experiments/H1.md`](experiments/H1.md) for the complete result and limitation.

## Experiment H2: nonuniform anatomy and wound repair

H2 replaces the uniform tube with seven visibly different anatomical regions:
tapered limbs, a thin bridge, a terminal bulb, and an off-axis pad. It then
removes a 29-cell patch from the bulb, feeds exactly 29 fresh cells from below,
requires those cells to mature locally, and finally articulates the repaired
body.

Strictly local deficit following failed because the feed source was captured by
nearer healthy tissue. The successful arm therefore gives each cell a
persistent **coarse anatomical-region guide**, chosen from live regional
capacity, while leaving its exact position continuous and unassigned. Broad
regional attraction brings a cell into the correct territory; local deficit
and density pressure then organize it. No cell receives a target-site index or
final coordinate.

The frozen configuration passes every predeclared gate on seeds 7, 19, and 31.
Across those runs, pre-wound coverage is 0.898–0.922; wound coverage falls to
0.069–0.138 after deletion and returns to 0.897–0.966; 93.1–100% of fresh cells
localize to the wound; and moving coverage remains 0.878–0.882. Nearest-bone,
pressure-off, and first-contact-material controls each fail for the expected
reason. See [`experiments/H2.md`](experiments/H2.md) for the complete trial
history and controls.

H2 is a meaningful developmental-controller result, but not yet a learned NCA:
the neural residual remains identically zero. The regional guide is also real
privileged structure, albeit far weaker than H0's one-site-per-cell assignment.

## Experiment H3: fine humanoid envelope and learned fate

H3 replaces the H with a connected 15-bone humanoid precursor and 1,092 target
cells across 18 anatomical regions. The tissue plan includes torso, head,
hands, feet, cheeks, and buttocks in addition to tapered bone capsules. Fully
45.0% of the reference volume lies more than 0.18 world units from its nearest
bone, so this cannot be represented as a uniform bone offset.

Spacing falls from H2's 0.14 to 0.10. The largest H3 material splat has world
radius 0.0195—69.6% of H2's 0.028—and regional splat scale varies from 0.80 to
1.30. Particle attachment state and rendering now support arbitrary bone
counts and material-dependent radii.

The primary learned component is a shared regional-fate MLP. It sees live
regional shortage, availability, and cell-to-region distance, then commits a
cell to one coarse anatomical fate; local deficit and pressure still determine
its continuous position. It achieves 98.4–99.7% held-out agreement with the
frozen capacity/distance oracle. The learned arm passes every development,
remote-cheek-repair, and motion gate on seeds 7, 19, and 31. Strictly local
deficit and a shortage-blind version of the same model both fail badly. See
[`experiments/H3.md`](experiments/H3.md).

This is the first humanoid substrate and the first learned developmental
decision, but it is still oracle distillation with a global regional-demand
vector. The acceleration residual remains zero, and the procedural 15-bone rig
is a precursor rather than a production human skeleton.

## Experiment H4: production rig and variable-thickness target

H4 ingests the supplied Meshy character through Blender and exports a portable
24-bone rig asset containing the bind mesh, UVs, weights, per-frame skin
matrices, dynamic bone endpoints, and Blender-evaluated walk vertices. Full
linear skinning reproduces Blender at `1.10e-7 m` RMS. Measured influence
ablations reject top 3 and top 4; top 6 passes at 0.106 mm RMS and is the frozen
persistent state.

The non-watertight source surface is voxelized and flood-filled into a single
13,273-cell target at 0.025 m pitch. It has no enclosed pocket, and 62.0% of its
mass lies more than 0.08 m from every bone. The maximum material-scaled splat
radius is 0.008625 m, less than half H3's maximum. This provides real head,
torso, cheek, pelvic, and gluteal volume instead of bone-centered capsules. See
[`experiments/H4.md`](experiments/H4.md).

H4 remains a kinematic baseline: nearest-surface state transfer is privileged,
and the source animation contains no elasticity or secondary flesh motion.

## Experiment H5: separately taught local flesh mechanics

H5 adds an explicit, under-damped graph-elastic teacher around H4's exact LBS
trajectory, then distills it into one shared 17-input local message MLP. The
rule sees residual position/velocity, skeleton acceleration, bone distance,
local stiffness, and mean six-neighbor residual/velocity differences; it never
sees cell identity, bone identity, phase, or a target position.

The final 2,400-step configuration passes every gate on seeds 7, 19, and 31.
Three-cycle free-rollout error is 0.478-0.528 mm RMS, 1.879-2.178 mm p99, and
16.051-19.619 mm maximum. Removing only the neighbor messages makes positional
error about 4x worse and edge-strain error 5.3-5.9x worse, demonstrating that
local transport is causally used. An earlier 1,200-step frozen run is retained
as a failure because one rare degree-2 surface cell exceeded the maximum-error
gate; the gate was not relaxed. See [`experiments/H5.md`](experiments/H5.md).

H5 teaches a controlled synthetic elastic behavior, not anatomy. Novel-motion,
novel-body, density, collision, incompressibility, and muscle behavior remain
open curricula.

## Experiment H5D: high-density scaling

H5D reduces pitch to 17.5 mm, increasing the population from 13,273 to 35,527
cells while leaving the shared rule at the same parameter count. Graph coupling
is scaled by inverse pitch squared to preserve continuum stiffness, and batch
size is scaled with population to preserve per-cell exposure.

The first seed-7 qualification passes without tuning, followed by a clean
three-seed pass. Median rollout RMS improves 17.6% to 0.423 mm; p99 is
1.612-1.827 mm and maximum error is 13.810-20.278 mm. The neighbor-blind arm is
4.3-4.9x worse in position and 6.8-7.5x worse in edge strain. Runtime grows
only 2.08x for 2.68x as many cells, and each checkpoint remains about 48 KiB.
See [`experiments/H5D.md`](experiments/H5D.md).

## Repository layout

```text
src/flesh_and_bone/  scaffold, morphology, particles, dynamics, metrics, render
scripts/             reproducible experiment entry points
tests/               CPU-fast geometry and dynamics contracts
experiments/         immutable run artifacts and research notes
```

Every source file has a companion Markdown file documenting its purpose and
contracts. `experiments/README.md` is the chronological research ledger.

## Running the experiments

Python 3.11+ with PyTorch, NumPy, and Pillow is required.

```bash
python -m pip install -e .
python scripts/run_h_baseline.py --device cpu
python scripts/run_h1.py --device cpu --arm all
python scripts/run_h2.py --device cpu --arm all
python scripts/run_h3.py --device cpu --arm all
python scripts/run_h4.py
python scripts/run_h5.py --device cuda
```

Apple Silicon can use `--device mps`; CUDA can use `--device cuda`. The runner
uses deterministic seeds and records the resolved configuration in its JSON.

## Research ladder

1. **H0 — mechanical control (complete):** feeding, assignment, checker
   differentiation, density pressure, articulated tracking, and preserved
   negative space.
2. **H1 — continuous field (complete):** remove explicit niche assignments;
   cells follow local tissue-deficit and density fields, then acquire bone
   coordinates.
3. **H2 — nonuniform anatomy and repair (complete):** replace uniform tubes with
   regional density/radius structure, erase a body region, and recruit fresh
   cells without exact-site assignments.
4. **H3 — humanoid learned fate (complete):** generalize to 15 bones, finer and
   variable splats, extra-skeletal tissue, and a learned global fate scorer.
5. **H4 — production human target (complete):** import a 24-bone rig, validate
   six-weight surface transport, and build a fine variable-thickness volume.
6. **H5 — learned local flesh (complete):** drive a graph-elastic teacher from
   the H4 bones and distill stable local secondary motion with a causal
   neighbor-message control.
7. **H5D — high-density scaling (complete):** increase the same body to 35,527
   cells while preserving the physical regime and learned-rule accuracy.
8. **H6 — pose/edge MoE:** pose experts stabilize tissue states; directed edge
   experts control difficult skeletal transitions under a hard global guide.
9. **H7 — scale:** bootstrap a 64³-equivalent splat creature from a mature,
   validated 32³-equivalent state rather than relearning assembly and animation
   simultaneously.

The project should reject the particle representation if stable density,
negative-space preservation, and articulated tracking require effectively
hard-coding every final particle position. In that case a dense Eulerian field
or hybrid particle-in-cell model is the better substrate.
