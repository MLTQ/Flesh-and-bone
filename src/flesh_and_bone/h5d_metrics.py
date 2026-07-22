"""Cross-discretization acceptance metrics for the H5D density trial."""

from statistics import median


def evaluate_h5d(baseline_h4, dense_h4, baseline_h5, dense_h5):
    """Compare validated H4/H5 reports and apply H5D's declared gates."""
    baseline_volume = baseline_h4["volume"]
    dense_volume = dense_h4["volume"]
    baseline_teacher = baseline_h5["teacher"]
    dense_teacher = dense_h5["teacher"]
    runs = dense_h5["runs"]

    cell_ratio = (
        dense_volume["cell_count"] / baseline_volume["cell_count"]
    )
    residual_ratio = (
        dense_teacher["residual_rms"]
        / baseline_teacher["residual_rms"]
    )
    softness_ratio = (
        dense_teacher["far_near_amplitude_ratio"]
        / baseline_teacher["far_near_amplitude_ratio"]
    )
    rollout_rms = [
        run["learned_rollout"]["position_rms"] for run in runs
    ]
    rollout_p99 = [
        run["learned_rollout"]["position_p99"] for run in runs
    ]
    rollout_max = [
        run["learned_rollout"]["position_max"] for run in runs
    ]
    metrics = {
        "baseline_cell_count": baseline_volume["cell_count"],
        "dense_cell_count": dense_volume["cell_count"],
        "cell_count_ratio": cell_ratio,
        "dense_pitch": dense_volume["pitch"],
        "teacher_residual_ratio": residual_ratio,
        "teacher_softness_ratio": softness_ratio,
        "median_rollout_rms": median(rollout_rms),
        "maximum_rollout_p99": max(rollout_p99),
        "maximum_rollout_error": max(rollout_max),
    }
    gates = {
        "dense_h4_pass": bool(dense_h4["acceptance"]["pass"]),
        "cell_count": cell_ratio >= 2.5,
        "pitch": abs(dense_volume["pitch"] - 0.0175) <= 1e-12,
        "connected": dense_volume["occupied_component_count"] == 1,
        "no_enclosed_pocket": (
            dense_volume["largest_enclosed_empty_pocket"] == 0
        ),
        "normalized_weights": (
            dense_volume["weight_sum_max_error"] <= 1e-6
        ),
        "finite_motion": bool(dense_volume["finite"]),
        "small_splats": (
            dense_volume["maximum_world_splat_radius"] <= 0.0061
        ),
        "teacher_residual_continuity": 0.85 <= residual_ratio <= 1.15,
        "teacher_softness_continuity": 0.85 <= softness_ratio <= 1.15,
        "inherited_h5": bool(dense_h5["acceptance"]["pass"]),
        "median_rollout_rms": median(rollout_rms) <= 0.00075,
        "rollout_p99": max(rollout_p99) <= 0.003,
        "rollout_max": max(rollout_max) <= 0.040,
    }
    return {"metrics": metrics, "gates": {**gates, "pass": all(gates.values())}}
