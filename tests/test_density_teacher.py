"""CPU-fast contracts for H7's nonlinear density observation and cap."""

from types import SimpleNamespace

import torch

from flesh_and_bone.density_teacher import (
    DensityTeacherConfig,
    density_observation,
    smooth_norm_cap,
)


def test_compressed_pair_receives_opposite_outward_vectors():
    graph = SimpleNamespace(
        source=torch.tensor([0, 1]),
        target=torch.tensor([1, 0]),
        degree=torch.ones(2),
    )
    lbs = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    residual = torch.tensor([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
    observation = density_observation(
        lbs, residual, graph, pitch=1.0, config=DensityTeacherConfig()
    )
    torch.testing.assert_close(
        observation.compression_rms, torch.full((2,), 0.15)
    )
    assert observation.stretch_rms.tolist() == [0.0, 0.0]
    assert observation.compression_vector[0, 0] < 0
    assert observation.compression_vector[1, 0] > 0
    torch.testing.assert_close(
        observation.compression_vector[0],
        -observation.compression_vector[1],
    )


def test_smooth_norm_cap_preserves_direction_and_stays_bounded():
    vector = torch.tensor([[30.0, 40.0, 0.0], [0.0, 0.0, 0.0]])
    capped = smooth_norm_cap(vector, 12.0)
    assert capped[0].norm() < 12.0
    torch.testing.assert_close(
        capped[0] / capped[0].norm(), vector[0] / vector[0].norm()
    )
    assert torch.equal(capped[1], vector[1])
