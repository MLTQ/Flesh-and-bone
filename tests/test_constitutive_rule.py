"""CPU-fast contracts for H6C's physical basis and identifier."""

from types import SimpleNamespace

import torch

from flesh_and_bone.constitutive_rule import (
    ConstitutiveFleshRule,
    constitutive_terms,
    fit_constitutive_rule,
)
from flesh_and_bone.flesh_rule import flesh_features


def test_constitutive_terms_and_rule_preserve_vector_axes():
    vector = torch.ones(2, 3)
    scalar = torch.full((2, 1), 4.0)
    features = flesh_features(
        vector,
        2 * vector,
        3 * vector,
        4 * vector,
        5 * vector,
        torch.zeros_like(scalar),
        scalar,
    )
    terms = constitutive_terms(features)
    assert terms.shape == (2, 5, 3)
    rule = ConstitutiveFleshRule(torch.ones(5))
    expected = torch.tensor(-4.0 - 4.0 + 4.0 + 5.0 - 3.0)
    assert torch.allclose(rule(features), expected.expand(2, 3))


def test_identifier_recovers_known_five_term_rule_from_phase_holdout():
    torch.manual_seed(4)
    phases, substeps, cells = 10, 2, 16
    shape = (phases, substeps, cells, 3)
    residual = torch.randn(shape) * 0.01
    velocity = torch.randn(shape) * 0.1
    neighbor_residual = torch.randn(shape) * 0.005
    neighbor_velocity = torch.randn(shape) * 0.03
    lbs_acceleration = torch.randn(phases, cells, 3)
    stiffness = torch.linspace(2.0, 8.0, cells)
    coefficients = torch.tensor([1.2, 0.7, 9.0, -0.2, 1.1])
    acceleration = torch.empty_like(residual)
    bone_distance = torch.linspace(0.0, 0.2, cells)
    for phase in range(phases):
        for substep in range(substeps):
            features = flesh_features(
                residual[phase, substep],
                velocity[phase, substep],
                lbs_acceleration[phase],
                neighbor_residual[phase, substep],
                neighbor_velocity[phase, substep],
                bone_distance[:, None],
                stiffness[:, None],
            )
            terms = constitutive_terms(features)
            acceleration[phase, substep] = torch.einsum(
                "nkj,k->nj", terms, coefficients
            )
    trajectory = SimpleNamespace(
        residual=residual,
        velocity=velocity,
        acceleration=acceleration,
        neighbor_residual=neighbor_residual,
        neighbor_velocity=neighbor_velocity,
        lbs_acceleration=lbs_acceleration,
        stiffness=stiffness,
    )
    volume = SimpleNamespace(bone_distance=bone_distance)
    _, fit = fit_constitutive_rule(trajectory, volume)
    assert torch.allclose(
        torch.tensor(fit.coefficients), coefficients, atol=2e-5, rtol=2e-5
    )
    assert fit.holdout_acceleration_nrmse < 1e-5
