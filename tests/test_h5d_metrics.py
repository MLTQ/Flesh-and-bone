"""CPU-fast contracts for H5D cross-discretization acceptance."""

from copy import deepcopy

from flesh_and_bone.h5d_metrics import evaluate_h5d


def _reports():
    baseline_h4 = {"volume": {"cell_count": 100}}
    dense_volume = {
        "cell_count": 270,
        "pitch": 0.0175,
        "occupied_component_count": 1,
        "largest_enclosed_empty_pocket": 0,
        "weight_sum_max_error": 1e-7,
        "finite": True,
        "maximum_world_splat_radius": 0.006,
    }
    dense_h4 = {"volume": dense_volume, "acceptance": {"pass": True}}
    baseline_h5 = {
        "teacher": {
            "residual_rms": 0.02,
            "far_near_amplitude_ratio": 2.0,
        }
    }
    run = {
        "learned_rollout": {
            "position_rms": 0.0005,
            "position_p99": 0.002,
            "position_max": 0.02,
        }
    }
    dense_h5 = {
        "teacher": {
            "residual_rms": 0.019,
            "far_near_amplitude_ratio": 1.9,
        },
        "runs": [deepcopy(run) for _ in range(3)],
        "acceptance": {"pass": True},
    }
    return baseline_h4, dense_h4, baseline_h5, dense_h5


def test_h5d_accepts_valid_density_scaling():
    assert evaluate_h5d(*_reports())["gates"]["pass"]


def test_h5d_rejects_density_without_continuum_teacher_match():
    reports = _reports()
    reports[3]["teacher"]["residual_rms"] = 0.03
    result = evaluate_h5d(*reports)
    assert not result["gates"]["teacher_residual_continuity"]
    assert not result["gates"]["pass"]
