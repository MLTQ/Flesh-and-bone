"""Predeclared H4 provenance, skinning, surface, and volume gates."""


def influence_pass(metrics):
    """Apply H4's frozen deployed-influence transport thresholds."""
    return (
        metrics["animation_rms"] <= 5e-4
        and metrics["animation_p99"] <= 2e-3
        and metrics["animation_max"] <= 1e-2
        and metrics["finite"]
    )


def select_influence_count(results):
    """Choose the smallest measured arm passing transport thresholds."""
    passing = sorted(
        result["influence_count"]
        for result in results if influence_pass(result)
    )
    return passing[0] if passing else None


def acceptance_h4(provenance, full_skinning, selected_skinning, surface,
                  volume):
    """Apply all predeclared H4 asset/transport/target gates."""
    gates = {
        "source_members": provenance["source_members_recorded"],
        "topology_match": provenance["topology_match"],
        "weighted_bones": provenance["weighted_bone_count"] >= 20,
        "no_unweighted_vertices": provenance["unweighted_vertex_count"] == 0,
        "normalized_source_weights": (
            provenance["weight_sum_max_error"] <= 2e-5
        ),
        "uv_present": provenance["uv_layer_count"] >= 1,
        "canonical_height": 1.5 <= provenance["height"] <= 2.0,
        "canonical_ground": abs(provenance["ground"]) <= 0.02,
        "rest_round_trip": provenance["rest_pose_rms"] <= 1e-6,
        "full_skin_rms": full_skinning["animation_rms"] <= 1e-5,
        "full_skin_p99": full_skinning["animation_p99"] <= 5e-5,
        "full_skin_max": full_skinning["animation_max"] <= 2e-4,
        "animation_loop": provenance["loop_skin_matrix_rms"] <= 1e-4,
        "selected_influences": influence_pass(selected_skinning),
        "surface_complete": surface["cell_count"] == provenance["vertex_count"],
        "surface_finite": surface["finite"],
        "volume_cell_count": 5000 <= volume["cell_count"] <= 50000,
        "volume_pitch": volume["pitch"] <= 0.025,
        "volume_connected": volume["occupied_component_count"] == 1,
        "volume_pockets": volume["enclosed_pocket_fraction"] <= 0.005,
        "volume_extra_skeletal": volume["extra_skeletal_fraction"] >= 0.10,
        "volume_bone_distance": volume["bone_distance_p95"] >= 0.12,
        "volume_small_splats": (
            volume["maximum_world_splat_radius"] <= 0.0195
        ),
        "volume_normalized_weights": volume["weight_sum_max_error"] <= 2e-5,
        "volume_valid_bones": volume["valid_dominant_bone_fraction"] == 1.0,
        "volume_finite": volume["finite"],
    }
    return {**gates, "pass": all(gates.values())}
