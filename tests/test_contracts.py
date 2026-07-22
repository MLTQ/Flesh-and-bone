"""CPU-fast contracts for the H scaffold, body plan, and particle control."""

import torch

from flesh_and_bone.deficit_dynamics import DeficitDynamics
from flesh_and_bone.dynamics import MechanicalDynamics, ParticleNCARule
from flesh_and_bone.metrics import measure_state
from flesh_and_bone.morphology import (
    build_h_body_plan,
    checker_at_points,
    deform_body_plan,
)
from flesh_and_bone.particles import ParticleSystem
from flesh_and_bone.skeleton import HScaffold, bone_frames


def test_h_has_two_connected_degree_three_junctions():
    frame = HScaffold().frame(0)
    left = frame.endpoints[2, 0]
    right = frame.endpoints[2, 1]
    torch.testing.assert_close(frame.endpoints[0, 0], left)
    torch.testing.assert_close(frame.endpoints[1, 0], left)
    torch.testing.assert_close(frame.endpoints[3, 0], right)
    torch.testing.assert_close(frame.endpoints[4, 0], right)


def test_bone_frames_are_orthonormal():
    frame = HScaffold().frame(1.7)
    _, lengths, basis = bone_frames(frame.endpoints)
    identity = torch.eye(3).expand(5, -1, -1)
    torch.testing.assert_close(
        basis.transpose(1, 2) @ basis, identity, atol=1e-5, rtol=1e-5
    )
    assert (lengths > 0.5).all()


def test_body_plan_has_checker_skin_and_preserves_rest_embedding():
    scaffold = HScaffold()
    body = build_h_body_plan(scaffold, spacing=0.20, radius=0.24)
    sites, gaps = deform_body_plan(body, scaffold.frame(0))
    assert 80 < body.site_count < 500
    assert set(body.checker.tolist()) == {0, 1}
    torch.testing.assert_close(sites, body.reference_sites, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        checker_at_points(sites, body.checker_origin, body.checker_size),
        body.checker,
    )
    assert gaps.shape == (2, 3)
    torch.testing.assert_close(body.bone_weights.sum(1), torch.ones(body.site_count))


def test_feed_reserves_unique_sites_and_commitment_is_persistent():
    scaffold = HScaffold()
    body = build_h_body_plan(scaffold, spacing=0.22, radius=0.23)
    targets, _ = deform_body_plan(body, scaffold.frame(0))
    particles = ParticleSystem(body.site_count + 4)
    generator = torch.Generator().manual_seed(4)
    fed = particles.feed(body, targets, 12, generator, source=(0, 0, 0), jitter=0)
    assigned = particles.assigned_site[particles.active]
    assert fed == 12 and assigned.unique().numel() == 12
    particles.positions[particles.active] = targets[assigned]
    assert particles.update_commitment(body, targets, threshold=0.01) == 12
    previous = particles.checker.clone()
    particles.positions[particles.active] += 1
    particles.update_commitment(body, targets, threshold=0.01)
    torch.testing.assert_close(particles.checker, previous)


def test_zero_initialized_particle_nca_is_exactly_inert():
    rule = ParticleNCARule()
    output = rule(torch.randn(9, rule.input_channels))
    torch.testing.assert_close(output, torch.zeros_like(output), atol=0, rtol=0)


def test_short_control_reduces_tracking_error_and_stays_finite():
    scaffold = HScaffold()
    body = build_h_body_plan(scaffold, spacing=0.23, radius=0.22)
    targets, gaps = deform_body_plan(body, scaffold.frame(0))
    particles = ParticleSystem(body.site_count)
    generator = torch.Generator().manual_seed(3)
    particles.feed(body, targets, body.site_count, generator, source=(0, -1.2, 0))
    before = measure_state(particles, body, targets, gaps)
    dynamics = MechanicalDynamics(body)
    with torch.no_grad():
        for _ in range(90):
            dynamics.step(particles, targets, 0)
    after = measure_state(particles, body, targets, gaps)
    assert after["mean_tracking_error"] < before["mean_tracking_error"]
    assert after["finite"] and after["mass"] == after["active"]


def test_continuous_commitment_never_exposes_target_site_identity():
    scaffold = HScaffold()
    frame = scaffold.frame(0)
    body = build_h_body_plan(scaffold, spacing=0.22, radius=0.23)
    targets, _ = deform_body_plan(body, frame)
    particles = ParticleSystem(8)
    generator = torch.Generator().manual_seed(5)
    assert particles.feed_unassigned(8, generator, jitter=0) == 8
    particles.positions[particles.active] = targets[:8]
    assert particles.update_continuous_commitment(
        body, frame, targets, threshold=0.01
    ) == 8
    assert (particles.assigned_site == -1).all()
    assert particles.committed.all()
    assert not particles.material_locked.any()
    reconstructed = particles.attachment_targets(frame)
    torch.testing.assert_close(
        reconstructed, particles.positions[particles.active], atol=2e-5, rtol=2e-5
    )


def test_short_deficit_control_recruits_unassigned_cells_toward_tissue():
    scaffold = HScaffold()
    frame = scaffold.frame(0)
    body = build_h_body_plan(scaffold, spacing=0.22, radius=0.23)
    targets, _ = deform_body_plan(body, frame)
    particles = ParticleSystem(32)
    generator = torch.Generator().manual_seed(6)
    particles.feed_unassigned(32, generator, source=(0, -1.25, 0))
    before = torch.cdist(
        particles.positions[particles.active], targets
    ).min(dim=1).values.mean()
    dynamics = DeficitDynamics(body)
    with torch.no_grad():
        for _ in range(90):
            dynamics.step(particles, targets, frame, 0, moving=False)
    after = torch.cdist(
        particles.positions[particles.active], targets
    ).min(dim=1).values.mean()
    assert after < before
    assert (particles.assigned_site == -1).all()
    assert torch.isfinite(particles.positions).all()


def test_plastic_checker_tracks_body_field_until_material_lock():
    scaffold = HScaffold()
    frame = scaffold.frame(0)
    body = build_h_body_plan(scaffold, spacing=0.22, radius=0.23)
    targets, _ = deform_body_plan(body, frame)
    particles = ParticleSystem(1)
    generator = torch.Generator().manual_seed(8)
    particles.feed_unassigned(1, generator, jitter=0)
    first = 0
    opposite = int(torch.nonzero(body.checker != body.checker[first])[0].item())
    particles.positions[0] = targets[first]
    assert particles.update_continuous_commitment(
        body, frame, targets, threshold=0.01
    ) == 1
    particles.positions[0] = targets[opposite]
    particles.refresh_continuous_attachments(body, frame)
    particles.refresh_plastic_material(body, targets)
    assert particles.checker[0] == body.checker[opposite]
    locked_checker = particles.checker[0].clone()
    assert particles.lock_material() == 1
    particles.positions[0] = targets[first]
    particles.refresh_continuous_attachments(body, frame)
    particles.refresh_plastic_material(body, targets)
    assert particles.checker[0] == locked_checker
    assert particles.material_locked[0]


def test_checker_field_metric_detects_spatially_inverted_texture():
    scaffold = HScaffold()
    frame = scaffold.frame(0)
    body = build_h_body_plan(scaffold, spacing=0.23, radius=0.22)
    targets, gaps = deform_body_plan(body, frame)
    particles = ParticleSystem(body.site_count)
    particles.active[:] = True
    particles.mass[:] = 1
    particles.positions[:] = targets
    particles.committed[:] = True
    particles.material_locked[:] = True
    particles.part[:] = body.dominant_bone
    particles.checker[:] = body.checker
    correct = measure_state(particles, body, targets, gaps)
    assert correct["checker_field_accuracy"] == 1.0
    particles.checker[:] = 1 - body.checker
    inverted = measure_state(particles, body, targets, gaps)
    assert inverted["checker_field_accuracy"] == 0.0
