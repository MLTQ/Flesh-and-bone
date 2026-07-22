"""Region, wound-repair, lineage, and H2 acceptance measurements."""

import torch

from .metrics import measure_state
from .skeleton import segment_projection


def measure_h2(particles, body_plan, target_positions, gap_positions, frame):
    """Measure nonuniform anatomy and generation-1 wound repair."""
    base = measure_state(
        particles, body_plan, target_positions, gap_positions
    )
    region_count = len(body_plan.region_names)
    indices, positions, _, _ = particles.active_tensors()
    if not indices.numel():
        return {
            **base,
            "region_coverage": [0.0] * region_count,
            "minimum_region_coverage": 0.0,
            "region_allocation_error": 1.0,
            "wound_coverage": 0.0,
            "healthy_coverage": 0.0,
            "radial_profile_error": float("inf"),
            "repair_cells": 0,
            "guided_cells": 0,
            "repair_wound_localization": 0.0,
            "repair_commitment_fraction": 0.0,
            "repair_lock_fraction": 0.0,
        }

    site_distance = torch.cdist(target_positions, positions)
    nearest_distance, nearest_particle = site_distance.min(dim=1)
    covered = nearest_distance <= 0.78 * body_plan.spacing
    region_coverage = []
    for region in range(region_count):
        selected = body_plan.region == region
        region_coverage.append(float(
            covered[selected].float().mean().item() if selected.any() else 0.0
        ))

    _, nearest_site = site_distance.min(dim=0)
    actual_region = torch.bincount(
        body_plan.region[nearest_site], minlength=region_count
    ).to(positions.dtype)
    actual_region = actual_region / actual_region.sum().clamp(min=1)
    desired_region = torch.bincount(
        body_plan.region, minlength=region_count
    ).to(positions.dtype)
    desired_region = desired_region / desired_region.sum().clamp(min=1)
    region_allocation_error = 0.5 * (
        actual_region - desired_region
    ).abs().sum()

    wound = body_plan.wound_mask
    healthy = ~wound
    wound_coverage = covered[wound].float().mean()
    healthy_coverage = covered[healthy].float().mean()
    _, _, bone_distance = segment_projection(positions, frame.endpoints)
    actual_bone_distance = bone_distance.amin(dim=1)
    radial_profile_error = (
        actual_bone_distance
        - body_plan.reference_bone_distance[nearest_site]
    ).abs().mean()

    repair_indices = indices[particles.generation[indices] == 1]
    if repair_indices.numel():
        repair_positions = particles.positions[repair_indices]
        wound_distance = torch.cdist(
            repair_positions, target_positions[wound]
        ).amin(dim=1)
        repair_wound_localization = (
            wound_distance <= 1.5 * body_plan.spacing
        ).float().mean()
        repair_commitment = particles.committed[repair_indices].float().mean()
        repair_lock = particles.material_locked[repair_indices].float().mean()
    else:
        repair_wound_localization = positions.new_zeros(())
        repair_commitment = positions.new_zeros(())
        repair_lock = positions.new_zeros(())

    return {
        **base,
        "region_coverage": region_coverage,
        "minimum_region_coverage": min(region_coverage),
        "region_allocation_error": float(region_allocation_error.item()),
        "wound_coverage": float(wound_coverage.item()),
        "healthy_coverage": float(healthy_coverage.item()),
        "radial_profile_error": float(radial_profile_error.item()),
        "repair_cells": int(repair_indices.numel()),
        "guided_cells": int((particles.guide_region[indices] >= 0).sum().item()),
        "repair_wound_localization": float(repair_wound_localization.item()),
        "repair_commitment_fraction": float(repair_commitment.item()),
        "repair_lock_fraction": float(repair_lock.item()),
    }


def acceptance_h2(pre_wound, damaged, repaired, moving, site_count):
    """Apply the predeclared H2 development, repair, and motion gates."""
    gates = {
        "development_coverage": pre_wound["coverage"] >= 0.85,
        "development_regions": pre_wound["minimum_region_coverage"] >= 0.70,
        "damage_effect": damaged["wound_coverage"] <= (
            pre_wound["wound_coverage"] - 0.30
        ),
        "wound_repair": (
            repaired["wound_coverage"] >= 0.78
            and repaired["wound_coverage"]
            >= pre_wound["wound_coverage"] - 0.12
        ),
        "healthy_retention": repaired["healthy_coverage"] >= (
            pre_wound["healthy_coverage"] - 0.08
        ),
        "repair_localization": repaired["repair_wound_localization"] >= 0.55,
        "repair_commitment": repaired["repair_commitment_fraction"] >= 0.90,
        "repair_maturation": repaired["repair_lock_fraction"] >= 0.90,
        "checker_repair": repaired["checker_field_accuracy"] >= 0.80,
        "checker_motion": moving["checker_field_accuracy"] >= 0.80,
        "region_allocation": repaired["region_allocation_error"] <= 0.16,
        "moving_coverage": moving["coverage"] >= 0.82,
        "tracking": moving["mean_tracking_error"] <= 0.12,
        "crowding": moving["crowding_fraction"] < 0.08,
        "negative_space": max(moving["gap_occupancy"]) <= 0.02,
        "restored_mass": moving["active"] == site_count,
        "mass_accounting": abs(moving["mass"] - moving["active"]) < 1e-5,
        "no_assignment_leakage": moving["exposed_assignments"] == 0,
        "finite": all(
            state["finite"] for state in (pre_wound, damaged, repaired, moving)
        ),
    }
    return {**gates, "pass": all(gates.values())}
