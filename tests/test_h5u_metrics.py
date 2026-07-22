"""CPU-fast contracts for H5U ultra-density acceptance."""

from copy import deepcopy

from flesh_and_bone.h5u_metrics import evaluate_h5u


def _reports():
    baseline_h4 = {"volume": {"cell_count": 100}}
    ultra_volume = {
        "cell_count": 700,
        "pitch": 0.0125,
        "occupied_component_count": 1,
        "largest_enclosed_empty_pocket": 0,
        "weight_sum_max_error": 1e-7,
        "finite": True,
        "uv_transfer": "closest-triangle-barycentric",
        "splat_scale_max": 1.15,
    }
    acceptance = {
        "source_members": True,
        "volume_cell_count": False,
        "pass": False,
    }
    ultra_h4 = {"volume": ultra_volume, "acceptance": acceptance}
    baseline_h5 = {
        "teacher": {
            "residual_rms": 0.02,
            "far_near_amplitude_ratio": 2.0,
        }
    }
    run = {
        "learned_rollout": {
            "position_rms": 0.0004,
            "position_p99": 0.0015,
            "position_max": 0.02,
        }
    }
    ultra_h5 = {
        "config": {
            "image_size": 720,
            "render_splat_radius_scale": 0.50,
            "render_opacity": 0.72,
        },
        "teacher": {
            "residual_rms": 0.019,
            "far_near_amplitude_ratio": 1.9,
        },
        "runs": [deepcopy(run) for _ in range(3)],
        "acceptance": {"pass": True},
    }
    return baseline_h4, ultra_h4, baseline_h5, ultra_h5


def test_h5u_accepts_only_the_declared_h4_scope_failure():
    assert evaluate_h5u(*_reports())["gates"]["pass"]


def test_h5u_rejects_nearest_vertex_uv_even_with_good_dynamics():
    reports = _reports()
    reports[1]["volume"]["uv_transfer"] = "nearest-vertex"
    result = evaluate_h5u(*reports)
    assert not result["gates"]["barycentric_uv"]
    assert not result["gates"]["pass"]
