"""CPU-fast feature-shift and gate contracts for H6M."""

from types import SimpleNamespace

import torch

from flesh_and_bone.flesh_rule import FleshResidualRule
from flesh_and_bone.h6m_metrics import (
    acceptance_h6m,
    measure_frozen_feature_shift,
)


def test_feature_shift_samples_without_refitting_rule_buffers():
    shape = (2, 2, 3, 3)
    vector = torch.zeros(shape)
    trajectory = SimpleNamespace(
        residual=vector,
        velocity=vector,
        neighbor_residual=vector,
        neighbor_velocity=vector,
        lbs_acceleration=torch.zeros(2, 3, 3),
        stiffness=torch.ones(3),
    )
    volume = SimpleNamespace(bone_distance=torch.zeros(3))
    rule = FleshResidualRule(hidden_channels=4)
    before = rule.feature_mean.clone()
    measured = measure_frozen_feature_shift(
        trajectory, volume, rule, sample_examples=7
    )
    assert measured["sample_examples"] == 7
    assert measured["z_abs_max"] == 1.0
    assert torch.equal(rule.feature_mean, before)


def test_acceptance_requires_neighbor_causality_without_relaxing_rollout():
    teacher = {
        "finite": True,
        "cycle_seam_rms": 0.0,
        "residual_rms": 0.01,
        "edge_difference_rms": 0.005,
    }
    learned = {
        "position_rms": 0.001,
        "position_p99": 0.002,
        "position_max": 0.003,
        "amplitude_ratio": 1.0,
        "far_near_amplitude_ratio": 2.0,
        "phase_zero_cycle_drift_rms": 0.0001,
        "finite": True,
        "lbs_improvement_fraction": 0.9,
        "edge_strain_error_rms": 0.001,
    }
    blind = {**learned, "position_rms": 0.0013}
    assert acceptance_h6m(teacher, learned, blind)["pass"]
    blind["position_rms"] = learned["position_rms"]
    assert not acceptance_h6m(teacher, learned, blind)["pass"]
