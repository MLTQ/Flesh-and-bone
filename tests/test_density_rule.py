"""CPU-fast contracts for H7's bounded coefficient learner."""

import torch

from flesh_and_bone.density_rule import BoundedDensityResidual


def test_density_coefficients_are_nonnegative_and_bounded():
    rule = BoundedDensityResidual(hidden_channels=8)
    scalars = torch.randn(64, 5) * 100
    coefficients = rule.coefficients(scalars)
    assert torch.all(coefficients >= 0)
    assert torch.all(coefficients[:, 0] <= 60)
    assert torch.all(coefficients[:, 1] <= 18)


def test_zero_initialized_logits_start_at_midpoint_law():
    rule = BoundedDensityResidual(hidden_channels=8)
    coefficients = rule.coefficients(torch.randn(5, 5))
    torch.testing.assert_close(
        coefficients, torch.tensor([30.0, 9.0]).expand(5, 2)
    )


def test_density_output_uses_explicit_directions_and_cap():
    rule = BoundedDensityResidual(hidden_channels=8)
    scalars = torch.zeros(2, 5)
    compression = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    stretch = torch.zeros_like(compression)
    acceleration = rule(scalars, compression, stretch)
    assert acceleration[0, 0] > 0 and acceleration[0, 1] == 0
    assert acceleration[1, 1] > 0 and acceleration[1, 0] == 0
    assert torch.all(acceleration.norm(dim=-1) < 12)
