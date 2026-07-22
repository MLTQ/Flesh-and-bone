"""CPU-fast contracts for H3 humanoid geometry and learned fate."""

import torch

from flesh_and_bone.fate_model import LearnedFateSelector, train_fate_model
from flesh_and_bone.h3_metrics import representation_gates
from flesh_and_bone.h3_morphology import build_h3_body_plan
from flesh_and_bone.humanoid_skeleton import HumanoidScaffold
from flesh_and_bone.particles import ParticleSystem


def test_humanoid_frame_and_extra_skeletal_morphology():
    scaffold = HumanoidScaffold()
    frame = scaffold.frame(0.0)
    body = build_h3_body_plan(scaffold)
    assert frame.endpoints.shape == (15, 2, 3)
    assert frame.bone_count == 15
    assert len(body.region_names) == 18
    assert torch.all(torch.bincount(body.region, minlength=18) > 0)
    assert int(body.wound_mask.sum()) >= 8
    assert (body.reference_bone_distance > 0.18).float().mean() >= 0.08
    assert body.splat_scale.max() / body.splat_scale.min() >= 1.5


def test_h3_representation_passes_predeclared_resolution_gates():
    scaffold = HumanoidScaffold()
    body = build_h3_body_plan(scaffold)
    values, gates = representation_gates(body, scaffold.bone_count, 0.15)
    assert gates["pass"]
    assert values["maximum_world_splat_radius"] < 0.20 * 0.14


def test_particle_reservoir_uses_dynamic_bones_and_material_splat_scale():
    scaffold = HumanoidScaffold()
    body = build_h3_body_plan(scaffold)
    particles = ParticleSystem(4, bone_count=scaffold.bone_count)
    generator = torch.Generator().manual_seed(5)
    particles.feed_unassigned(1, generator)
    particles.positions[0] = body.reference_sites[body.region == 10][0]
    targets = body.reference_sites
    particles.update_continuous_commitment(
        body, scaffold.frame(0.0), targets, threshold=0.01
    )
    assert particles.bone_weights.shape == (4, 15)
    assert particles.bone_local.shape == (4, 15, 3)
    assert particles.splat_scale[0] == body.splat_scale[body.region == 10][0]


def test_fate_model_learns_shared_oracle_score_and_ablation_changes_choice():
    model, report = train_fate_model(
        18, seed=13, steps=150, batch_size=256
    )
    assert report.holdout_agreement >= 0.90
    shortage = torch.zeros(18)
    shortage[4] = 10
    target = torch.full((18,), 10.0)
    distance = torch.linspace(0.1, 2.0, 18)
    learned = LearnedFateSelector(model, spacing=0.10, expose_shortage=True)
    blind = LearnedFateSelector(model, spacing=0.10, expose_shortage=False)
    assert learned(shortage, target, distance) == 4
    assert blind(shortage, target, distance) != 4
