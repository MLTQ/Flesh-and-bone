"""Acceptance measurements for particle assembly and articulated tracking."""

import torch


def measure_state(particles, body_plan, target_positions, gap_positions,
                  density_radius_scale=1.9):
    """Measure one particle state without changing the simulation."""
    indices, position, _, assignments = particles.active_tensors()
    if not indices.numel():
        return {
            "active": 0,
            "committed": 0,
            "material_locked": 0,
            "exposed_assignments": 0,
            "mass": 0.0,
            "coverage": 0.0,
            "mean_tracking_error": float("inf"),
            "density_relative_error": float("inf"),
            "crowding_fraction": 0.0,
            "gap_occupancy": [0.0, 0.0],
            "part_balance_error": 1.0,
            "checker_balance_error": 1.0,
            "checker_field_accuracy": 0.0,
            "finite": True,
        }

    distance_to_sites = torch.cdist(target_positions, position)
    nearest = distance_to_sites.amin(dim=1)
    coverage = (nearest <= 0.78 * body_plan.spacing).float().mean()
    if (assignments >= 0).all():
        tracking = (position - target_positions[assignments]).norm(dim=1)
        density_site = assignments
    else:
        tracking, density_site = distance_to_sites.min(dim=0)

    pair_distance = torch.cdist(position, position)
    identity = torch.eye(position.shape[0], device=position.device, dtype=torch.bool)
    density_radius = density_radius_scale * body_plan.spacing
    density_kernel = torch.exp(-(pair_distance / density_radius).square())
    density_kernel[identity] = 0
    density = density_kernel.sum(dim=1)
    target_density = body_plan.target_density[density_site]
    density_error = (
        (density - target_density).abs() / target_density.clamp(min=0.25)
    ).mean()
    pairs = torch.triu(~identity, diagonal=1)
    crowding = (
        pair_distance[pairs] < 0.55 * body_plan.spacing
    ).float().mean() if pairs.any() else position.new_zeros(())

    gap_radius = 0.72 * body_plan.radius
    gap_distance = torch.cdist(gap_positions, position)
    gap_occupancy = (
        gap_distance < gap_radius
    ).float().sum(dim=1) / max(position.shape[0], 1)
    finite = all(torch.isfinite(value).all().item() for value in (
        position, particles.velocities[indices], density
    ))
    committed_indices = indices[particles.committed[indices]]
    if committed_indices.numel():
        actual_parts = torch.bincount(
            particles.part[committed_indices], minlength=5
        ).to(position.dtype)
        actual_parts = actual_parts / actual_parts.sum()
        desired_parts = torch.bincount(
            body_plan.dominant_bone, minlength=5
        ).to(position.dtype)
        desired_parts = desired_parts / desired_parts.sum()
        part_balance_error = 0.5 * (actual_parts - desired_parts).abs().sum()
        actual_checker = particles.checker[committed_indices].float().mean()
        desired_checker = body_plan.checker.float().mean()
        checker_balance_error = (actual_checker - desired_checker).abs()
    else:
        part_balance_error = position.new_ones(())
        checker_balance_error = position.new_ones(())
    nearest_particle = distance_to_sites.argmin(dim=1)
    nearest_global = indices[nearest_particle]
    texture_valid = (
        nearest <= 0.78 * body_plan.spacing
    ) & particles.committed[nearest_global]
    if texture_valid.any():
        checker_field_accuracy = (
            particles.checker[nearest_global[texture_valid]]
            == body_plan.checker[texture_valid]
        ).float().mean()
    else:
        checker_field_accuracy = position.new_zeros(())
    return {
        "active": int(indices.numel()),
        "committed": int(particles.committed[indices].sum().item()),
        "material_locked": int(particles.material_locked[indices].sum().item()),
        "exposed_assignments": int((assignments >= 0).sum().item()),
        "mass": float(particles.mass[indices].sum().item()),
        "coverage": float(coverage.item()),
        "mean_tracking_error": float(tracking.mean().item()),
        "max_tracking_error": float(tracking.max().item()),
        "density_relative_error": float(density_error.item()),
        "crowding_fraction": float(crowding.item()),
        "gap_occupancy": [float(value) for value in gap_occupancy.tolist()],
        "part_balance_error": float(part_balance_error.item()),
        "checker_balance_error": float(checker_balance_error.item()),
        "checker_field_accuracy": float(checker_field_accuracy.item()),
        "finite": bool(finite),
    }


def acceptance(assembly, moving, site_count):
    """Return named H0 gates and the aggregate decision."""
    gates = {
        "coverage": moving["coverage"] >= 0.90,
        "tracking": moving["mean_tracking_error"] <= 0.10,
        "commitment": moving["committed"] >= int(0.95 * site_count),
        "crowding": moving["crowding_fraction"] < 0.08,
        "negative_space": max(moving["gap_occupancy"]) <= 0.02,
        "motion_retention": moving["coverage"] >= assembly["coverage"] - 0.08,
        "mass_accounting": abs(moving["mass"] - moving["active"]) < 1e-5,
        "finite": assembly["finite"] and moving["finite"],
    }
    return {**gates, "pass": all(gates.values())}


def acceptance_h1(assembly, moving, site_count):
    """Apply H1 gates including no assignment leakage and tissue balance."""
    gates = acceptance(assembly, moving, site_count)
    gates.pop("pass")
    gates.update({
        "density_homeostasis": moving["density_relative_error"] <= 0.20,
        "part_balance": moving["part_balance_error"] <= 0.15,
        "checker_balance": moving["checker_balance_error"] <= 0.10,
        "checker_field": moving["checker_field_accuracy"] >= 0.80,
        "material_lock": moving["material_locked"] == moving["committed"],
        "no_assignment_leakage": moving["exposed_assignments"] == 0,
    })
    return {**gates, "pass": all(gates.values())}
