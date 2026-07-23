"""CPU-fast contracts for H5 acceptance gates and causal controls."""

from copy import deepcopy

import torch

from flesh_and_bone.h5_metrics import acceptance_h5, flat_quantile


def _passing_inputs():
    teacher = {
        "graph_component_count": 1,
        "graph_min_degree": 1.0,
        "finite": True,
        "cycle_seam_rms": 1e-6,
        "residual_rms": 0.02,
        "far_near_amplitude_ratio": 2.0,
        "edge_difference_rms": 0.005,
    }
    training = {"holdout_acceleration_nrmse": 0.05}
    learned = {
        "position_rms": 0.001,
        "position_p99": 0.004,
        "position_max": 0.02,
        "amplitude_ratio": 1.0,
        "far_near_amplitude_ratio": 2.0,
        "phase_zero_cycle_drift_rms": 0.001,
        "finite": True,
        "edge_strain_error_rms": 0.0004,
        "lbs_improvement_fraction": 0.90,
    }
    neighbor_blind = {
        "position_rms": 0.0013,
        "edge_strain_error_rms": 0.0004,
    }
    return teacher, training, learned, neighbor_blind


def test_h5_acceptance_passes_only_with_causal_neighbor_degradation():
    inputs = _passing_inputs()
    assert acceptance_h5(*inputs)["pass"]

    blind = deepcopy(inputs[3])
    blind["position_rms"] = inputs[2]["position_rms"]
    result = acceptance_h5(inputs[0], inputs[1], inputs[2], blind)
    assert not result["neighbor_causal"]
    assert not result["pass"]


def test_h5_acceptance_does_not_hide_a_maximum_error_outlier():
    teacher, training, learned, neighbor_blind = _passing_inputs()
    learned["position_max"] = 0.040001
    result = acceptance_h5(teacher, training, learned, neighbor_blind)
    assert not result["rollout_max"]
    assert not result["pass"]


def test_flat_quantile_matches_torch_linear_interpolation():
    values = torch.tensor([9.0, 1.0, 4.0, 7.0, 2.0])
    assert torch.allclose(
        flat_quantile(values, 0.73),
        torch.quantile(values, 0.73),
    )
