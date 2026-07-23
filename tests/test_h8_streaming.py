from types import SimpleNamespace

import torch

from flesh_and_bone.constitutive_rule import ConstitutiveFleshRule
from flesh_and_bone.density_rule import BoundedDensityResidual, HybridDensityRule
from flesh_and_bone.density_teacher import DensityTeacherConfig
from flesh_and_bone.h8_streaming import (
    build_motion_stream,
    nonperiodic_acceleration,
    nonperiodic_resample,
    rollout_streaming_density,
    simulate_streaming_teacher,
)


def _small_body():
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    )
    volume = SimpleNamespace(
        points=points,
        cell_count=3,
        bone_distance=torch.tensor([0.01, 0.13, 0.20]),
        metadata={"pitch": 1.0},
    )
    graph = SimpleNamespace(
        source=torch.tensor([0, 1, 1, 2]),
        target=torch.tensor([1, 0, 2, 1]),
        degree=torch.tensor([1.0, 2.0, 1.0]),
    )
    return volume, graph


def _config():
    return DensityTeacherConfig(
        fps=10.0,
        substeps=1,
        warmup_cycles=0,
        near_stiffness=10.0,
        far_stiffness=10.0,
        softening_distance=1.0,
        damping_ratio=0.2,
        neighbor_coupling=1.0,
        pressure_near=0.0,
        pressure_far=0.0,
        cohesion_near=0.0,
        cohesion_far=0.0,
    )


def test_nonperiodic_resample_preserves_endpoints_and_hold_does_not_wrap():
    values = torch.arange(5, dtype=torch.float32)[:, None]
    resampled = nonperiodic_resample(values, 2.0)
    bones = values[:, None, None].expand(5, 1, 2, 1).repeat(1, 1, 1, 3)
    lbs = values[:, None].expand(5, 3, 1).repeat(1, 1, 3)
    stream = build_motion_stream(lbs, bones, 2.0, hold_frames=3, fps=10)

    assert resampled[:, 0].tolist() == [0.0, 2.0, 4.0]
    assert stream.motion_frames == 3
    assert stream.frame_count == 6
    assert torch.equal(stream.lbs_positions[-1], stream.lbs_positions[-2])
    assert not torch.equal(stream.lbs_positions[-1], stream.lbs_positions[0])


def test_nonperiodic_acceleration_never_uses_last_to_first_wrap():
    time = torch.arange(6, dtype=torch.float32)
    quadratic = (time.square())[:, None, None]
    acceleration = nonperiodic_acceleration(quadratic, fps=1.0)

    assert torch.allclose(acceleration, torch.full_like(acceleration, 2.0))


def test_cold_teacher_and_exact_backbone_stream_from_zero_state():
    volume, graph = _small_body()
    config = _config()
    base = volume.points[None].expand(8, -1, -1).clone()
    base[:, :, 1] += 0.01 * torch.arange(8)[:, None].square()
    bones = torch.zeros(8, 1, 2, 3)
    stream = build_motion_stream(base, bones, hold_frames=4, fps=10)
    teacher = simulate_streaming_teacher(
        stream, volume, graph, config, diagnostic_cells=3
    )
    backbone = ConstitutiveFleshRule(torch.tensor([1.0, 0.4, 1.0, 0.0, 1.0]))
    residual = BoundedDensityResidual(
        pressure_max=0.0, cohesion_max=0.0, acceleration_cap=12.0
    )
    hybrid = HybridDensityRule(backbone, residual).eval()
    rollout = rollout_streaming_density(
        hybrid, teacher, volume, graph, config, density_enabled=False
    )

    assert torch.equal(teacher.residual[0], torch.zeros_like(volume.points))
    assert torch.equal(rollout.residual[0], torch.zeros_like(volume.points))
    assert rollout.residual.shape == teacher.residual.shape
    assert torch.allclose(rollout.residual, teacher.residual, atol=1e-6)
    assert rollout.finite
