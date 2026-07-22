"""H3 humanoid, extra-skeletal, learned-fate, and repair evidence."""

import torch

from .h2_metrics import measure_h2


def measure_h3(particles, body_plan, target_positions, gap_positions, frame):
    """Extend repair evidence with critical lobes and guide allocation."""
    base = measure_h2(
        particles, body_plan, target_positions, gap_positions, frame
    )
    critical = [
        base["region_coverage"][index]
        for index in body_plan.critical_region_indices
    ]
    indices, positions, _, _ = particles.active_tensors()
    if indices.numel():
        distance = torch.cdist(target_positions, positions).amin(dim=1)
        extra = body_plan.reference_bone_distance > 0.18
        extra_coverage = float(
            (distance[extra] <= 0.78 * body_plan.spacing)
            .float().mean().item()
        )
        guide = particles.guide_region[indices]
        guided = guide >= 0
        if guided.any():
            region_count = len(body_plan.region_names)
            actual = torch.bincount(
                guide[guided], minlength=region_count
            ).to(positions.dtype)
            actual = actual / actual.sum().clamp(min=1)
            desired = torch.bincount(
                body_plan.region, minlength=region_count
            ).to(positions.dtype)
            desired = desired / desired.sum().clamp(min=1)
            guide_allocation_error = 0.5 * (actual - desired).abs().sum()
        else:
            guide_allocation_error = positions.new_ones(())
        active_scale = particles.splat_scale[indices]
        scale_min = float(active_scale.min().item())
        scale_max = float(active_scale.max().item())
    else:
        extra_coverage = 0.0
        guide_allocation_error = torch.tensor(1.0)
        scale_min = 0.0
        scale_max = 0.0
    return {
        **base,
        "critical_region_coverage": critical,
        "mean_critical_region_coverage": sum(critical) / len(critical),
        "extra_skeletal_coverage": extra_coverage,
        "guide_allocation_error": float(guide_allocation_error.item()),
        "active_splat_scale_min": scale_min,
        "active_splat_scale_max": scale_max,
    }


def representation_gates(body_plan, bone_count, base_splat_radius_scale):
    """Evaluate H3's predeclared resolution and anatomy representation claims."""
    maximum_world_radius = (
        base_splat_radius_scale
        * body_plan.spacing
        * float(body_plan.splat_scale.max().item())
    )
    scale_ratio = float(
        (body_plan.splat_scale.max() / body_plan.splat_scale.min()).item()
    )
    extra_fraction = float(
        (body_plan.reference_bone_distance > 0.18).float().mean().item()
    )
    values = {
        "bone_count": int(bone_count),
        "region_count": len(body_plan.region_names),
        "spacing": float(body_plan.spacing),
        "maximum_world_splat_radius": maximum_world_radius,
        "splat_scale_ratio": scale_ratio,
        "extra_skeletal_fraction": extra_fraction,
    }
    gates = {
        "bone_count": values["bone_count"] >= 15,
        "region_count": values["region_count"] >= 12,
        "fine_spacing": values["spacing"] <= 0.10 + 1e-8,
        "small_splats": maximum_world_radius <= 0.75 * (0.20 * 0.14),
        "variable_splats": scale_ratio >= 1.5,
        "extra_skeletal_volume": extra_fraction >= 0.08,
    }
    return values, {**gates, "pass": all(gates.values())}


def acceptance_h3_oracle(pre_wound, moving, representation):
    """Apply the predeclared fine-humanoid mechanical upper-bound gates."""
    gates = {
        "representation": representation["pass"],
        "development_coverage": pre_wound["coverage"] >= 0.80,
        "development_regions": pre_wound["minimum_region_coverage"] >= 0.55,
        "critical_lobes": pre_wound["mean_critical_region_coverage"] >= 0.70,
        "moving_coverage": moving["coverage"] >= 0.74,
        "tracking": moving["mean_tracking_error"] <= 0.12,
        "crowding": moving["crowding_fraction"] < 0.08,
        "checker_motion": moving["checker_field_accuracy"] >= 0.78,
        "no_assignment_leakage": moving["exposed_assignments"] == 0,
        "finite": pre_wound["finite"] and moving["finite"],
    }
    return {**gates, "pass": all(gates.values())}


def acceptance_h3_learned(pre_wound, damaged, repaired, moving, site_count,
                          representation, holdout_agreement,
                          oracle_pre_coverage):
    """Apply learned-fate development, remote repair, and motion gates."""
    gates = {
        "representation": representation["pass"],
        "learner_holdout": holdout_agreement >= 0.97,
        "development_coverage": pre_wound["coverage"] >= 0.78,
        "oracle_proximity": pre_wound["coverage"] >= oracle_pre_coverage - 0.06,
        "development_regions": pre_wound["minimum_region_coverage"] >= 0.50,
        "critical_lobes": pre_wound["mean_critical_region_coverage"] >= 0.65,
        "damage_effect": damaged["wound_coverage"] <= (
            pre_wound["wound_coverage"] - 0.30
        ),
        "wound_repair": (
            repaired["wound_coverage"] >= 0.70
            and repaired["wound_coverage"]
            >= pre_wound["wound_coverage"] - 0.18
        ),
        "healthy_retention": repaired["healthy_coverage"] >= (
            pre_wound["healthy_coverage"] - 0.10
        ),
        "repair_localization": repaired["repair_wound_localization"] >= 0.70,
        "repair_commitment": repaired["repair_commitment_fraction"] >= 0.85,
        "repair_maturation": repaired["repair_lock_fraction"] >= 0.85,
        "moving_coverage": moving["coverage"] >= 0.72,
        "tracking": moving["mean_tracking_error"] <= 0.13,
        "crowding": moving["crowding_fraction"] < 0.08,
        "checker_motion": moving["checker_field_accuracy"] >= 0.75,
        "all_guided": moving["guided_cells"] == site_count,
        "restored_mass": moving["active"] == site_count,
        "mass_accounting": abs(moving["mass"] - moving["active"]) < 1e-5,
        "no_assignment_leakage": moving["exposed_assignments"] == 0,
        "finite": all(
            state["finite"] for state in (
                pre_wound, damaged, repaired, moving
            )
        ),
    }
    return {**gates, "pass": all(gates.values())}
