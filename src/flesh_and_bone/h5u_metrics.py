"""Ultra-density, continuum, and render acceptance metrics for H5U."""

from statistics import median


def evaluate_h5u(baseline_h4, ultra_h4, baseline_h5, ultra_h5):
    """Apply H5U's predeclared cross-run and appearance-configuration gates."""
    baseline_volume = baseline_h4["volume"]
    ultra_volume = ultra_h4["volume"]
    baseline_teacher = baseline_h5["teacher"]
    ultra_teacher = ultra_h5["teacher"]
    config = ultra_h5["config"]
    runs = ultra_h5["runs"]

    failed_h4 = sorted(
        key for key, value in ultra_h4["acceptance"].items() if not value
    )
    cell_ratio = (
        ultra_volume["cell_count"] / baseline_volume["cell_count"]
    )
    residual_ratio = (
        ultra_teacher["residual_rms"]
        / baseline_teacher["residual_rms"]
    )
    softness_ratio = (
        ultra_teacher["far_near_amplitude_ratio"]
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
    world_radius = (
        config["render_splat_radius_scale"]
        * ultra_volume["pitch"]
        * ultra_volume["splat_scale_max"]
    )
    metrics = {
        "baseline_cell_count": baseline_volume["cell_count"],
        "ultra_cell_count": ultra_volume["cell_count"],
        "cell_count_ratio": cell_ratio,
        "ultra_pitch": ultra_volume["pitch"],
        "maximum_render_splat_radius": world_radius,
        "teacher_residual_ratio": residual_ratio,
        "teacher_softness_ratio": softness_ratio,
        "median_rollout_rms": median(rollout_rms),
        "maximum_rollout_p99": max(rollout_p99),
        "maximum_rollout_error": max(rollout_max),
        "h4_failed_gates": failed_h4,
    }
    gates = {
        "h4_scope_only": failed_h4 == ["pass", "volume_cell_count"],
        "cell_count": cell_ratio >= 6.5,
        "pitch": abs(ultra_volume["pitch"] - 0.0125) <= 1e-12,
        "connected": ultra_volume["occupied_component_count"] == 1,
        "no_enclosed_pocket": (
            ultra_volume["largest_enclosed_empty_pocket"] == 0
        ),
        "normalized_weights": (
            ultra_volume["weight_sum_max_error"] <= 1e-6
        ),
        "finite_motion": bool(ultra_volume["finite"]),
        "barycentric_uv": (
            ultra_volume.get("uv_transfer")
            == "closest-triangle-barycentric"
        ),
        "render_radius": (
            abs(config["render_splat_radius_scale"] - 0.50) <= 1e-12
            and world_radius <= 0.0072
        ),
        "render_opacity": abs(config["render_opacity"] - 0.72) <= 1e-12,
        "render_resolution": config["image_size"] == 720,
        "teacher_residual_continuity": 0.85 <= residual_ratio <= 1.15,
        "teacher_softness_continuity": 0.85 <= softness_ratio <= 1.15,
        "inherited_h5": bool(ultra_h5["acceptance"]["pass"]),
        "median_rollout_rms": median(rollout_rms) <= 0.00075,
        "rollout_p99": max(rollout_p99) <= 0.003,
        "rollout_max": max(rollout_max) <= 0.040,
    }
    return {
        "metrics": metrics,
        "gates": {**gates, "pass": all(gates.values())},
    }
