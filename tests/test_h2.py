"""CPU-fast contracts for H2 morphology, lineage, damage, and maturation."""

import torch

from flesh_and_bone.h2_metrics import measure_h2
from flesh_and_bone.h2_morphology import build_h2_body_plan
from flesh_and_bone.morphology import deform_body_plan
from flesh_and_bone.particles import ParticleSystem
from flesh_and_bone.skeleton import HScaffold


def test_h2_envelope_has_distinct_regions_and_nonuniform_radius():
    body = build_h2_body_plan(spacing=0.18)
    counts = torch.bincount(body.region, minlength=len(body.region_names))
    assert (counts > 0).all()
    assert 8 <= int(body.wound_mask.sum()) < int((body.region == 5).sum())
    bulb_radius = body.reference_bone_distance[body.region == 5].mean()
    thin_upper_radius = body.reference_bone_distance[body.region == 3].mean()
    pad_radius = body.reference_bone_distance[body.region == 6].mean()
    assert bulb_radius > thin_upper_radius + 0.06
    assert pad_radius > thin_upper_radius + 0.10


def test_wound_removal_clears_state_and_refeed_marks_lineage():
    particles = ParticleSystem(8)
    generator = torch.Generator().manual_seed(12)
    particles.feed_unassigned(6, generator, generation=0)
    particles.committed[:6] = True
    particles.material_locked[:6] = True
    assert particles.remove(torch.tensor([1, 3])) == 2
    assert particles.active_count == 4
    assert particles.generation[1] == -1
    assert not particles.committed[3]
    particles.feed_unassigned(2, generator, generation=1)
    assert particles.active_count == 6
    assert int((particles.generation == 1).sum()) == 2


def test_local_maturation_requires_consecutive_stability():
    scaffold = HScaffold()
    frame = scaffold.frame(0)
    body = build_h2_body_plan(scaffold, spacing=0.18)
    targets, _ = deform_body_plan(body, frame)
    particles = ParticleSystem(1)
    generator = torch.Generator().manual_seed(13)
    particles.feed_unassigned(1, generator, generation=1)
    particles.positions[0] = targets[0]
    particles.update_continuous_commitment(body, frame, targets, 0.01)
    indices = torch.tensor([0])
    assert particles.update_local_maturation(indices, torch.tensor([True]), 3) == 0
    assert particles.update_local_maturation(indices, torch.tensor([False]), 3) == 0
    for _ in range(2):
        assert particles.update_local_maturation(
            indices, torch.tensor([True]), 3
        ) == 0
    assert particles.update_local_maturation(
        indices, torch.tensor([True]), 3
    ) == 1
    assert particles.material_locked[0]


def test_h2_metrics_detect_predeclared_wound():
    scaffold = HScaffold()
    frame = scaffold.frame(0)
    body = build_h2_body_plan(scaffold, spacing=0.18)
    targets, gaps = deform_body_plan(body, frame)
    particles = ParticleSystem(body.site_count)
    particles.active[:] = True
    particles.mass[:] = 1
    particles.positions[:] = targets
    particles.committed[:] = True
    particles.material_locked[:] = True
    particles.generation[:] = 0
    particles.part[:] = body.dominant_bone
    particles.checker[:] = body.checker
    complete = measure_h2(particles, body, targets, gaps, frame)
    assert complete["coverage"] == 1.0
    assert complete["minimum_region_coverage"] == 1.0
    assert complete["wound_coverage"] == 1.0
    wound_particles = torch.nonzero(body.wound_mask).flatten()
    particles.remove(wound_particles)
    damaged = measure_h2(particles, body, targets, gaps, frame)
    assert damaged["wound_coverage"] < 0.5
    assert damaged["healthy_coverage"] > 0.9
