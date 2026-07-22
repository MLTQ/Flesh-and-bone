"""Continuous tissue-deficit recruitment without per-particle niche identities."""

from dataclasses import dataclass
import math

import torch

from .skeleton import segment_projection


@dataclass(frozen=True)
class DeficitDynamicsConfig:
    dt: float = 0.035
    attraction: float = 18.0
    attachment_attraction: float = 22.0
    moving_deficit_blend: float = 0.05
    damping: float = 4.5
    pressure: float = 6.0
    short_repulsion: float = 12.0
    occupancy_sigma_scale: float = 0.48
    recruitment_radius_scale: float = 1.3
    hierarchical_site_radius_scale: float = 3.0
    hierarchical_arrival_distance_scale: float = 2.5
    hierarchical_region_distance_weight: float = 0.04
    deficit_log_weight: float = 1.8
    deficit_floor: float = 0.015
    density_radius_scale: float = 1.9
    commit_distance_scale: float = 0.78
    maximum_speed: float = 2.4
    learned_acceleration_scale: float = 0.25
    maturation_speed_max: float = 0.16
    maturation_density_tolerance: float = 0.45
    maturation_distance_scale: float = 0.90
    maturation_required_steps: int = 18


class DeficitDynamics:
    """H1 field controller using current density deficits, never target indices."""

    def __init__(self, body_plan, config=None, pressure_enabled=True,
                 recruitment="deficit", plastic_material_enabled=True,
                 nearest_bone_uses_target_density=True,
                 region_selector=None):
        if recruitment not in {
            "deficit", "hierarchical_deficit", "nearest_bone"
        }:
            raise ValueError(
                "recruitment must be deficit, hierarchical_deficit, or nearest_bone"
            )
        self.body_plan = body_plan
        self.config = config or DeficitDynamicsConfig()
        self.pressure_enabled = bool(pressure_enabled)
        self.recruitment = recruitment
        self.plastic_material_enabled = bool(plastic_material_enabled)
        self.nearest_bone_uses_target_density = bool(
            nearest_bone_uses_target_density
        )
        self.region_selector = region_selector

    def step(self, particles, target_positions, frame, time, moving,
             learned_rule=None, allow_local_maturation=False):
        indices, position, velocity, _ = particles.active_tensors()
        if not indices.numel():
            return {
                "mean_site_deficit": 1.0,
                "unfilled_site_fraction": 1.0,
                "allocation_entropy": 1.0,
                "mean_density": 0.0,
                "mean_pressure": 0.0,
                "locally_matured": 0,
            }

        site_distance = torch.cdist(position, target_positions)
        occupancy_sigma = self.config.occupancy_sigma_scale * self.body_plan.spacing
        occupancy = torch.exp(-(site_distance / occupancy_sigma).square()).sum(dim=0)
        deficit = torch.relu(1 - occupancy)

        if self.recruitment in {"deficit", "hierarchical_deficit"}:
            if self.recruitment == "hierarchical_deficit":
                if not hasattr(self.body_plan, "region"):
                    raise ValueError(
                        "hierarchical deficit requires body-plan regions"
                    )
                region_count = len(self.body_plan.region_names)
                region_distance = torch.stack([
                    site_distance[:, self.body_plan.region == region].amin(dim=1)
                    for region in range(region_count)
                ], dim=1)
                selected_region = particles.guide_region[indices].clone()
                target_count = torch.bincount(
                    self.body_plan.region, minlength=region_count
                ).to(position.dtype)
                guided = selected_region[selected_region >= 0]
                guided_count = torch.bincount(
                    guided, minlength=region_count
                ).to(position.dtype)
                unselected = torch.nonzero(selected_region < 0).flatten()
                for local_index in unselected.tolist():
                    shortage = torch.relu(target_count - guided_count)
                    if self.region_selector is not None:
                        choice = self.region_selector(
                            shortage,
                            target_count,
                            region_distance[local_index],
                        )
                        if not 0 <= int(choice) < region_count:
                            raise ValueError("region selector returned invalid choice")
                        choice = int(choice)
                    elif shortage.sum() <= 0:
                        choice = region_distance[local_index].argmin()
                    else:
                        score = (
                            shortage / target_count.clamp(min=1)
                            - self.config.hierarchical_region_distance_weight
                            * region_distance[local_index]
                            / self.body_plan.spacing
                        )
                        score = score.masked_fill(shortage <= 0, -torch.inf)
                        choice = score.argmax()
                    selected_region[local_index] = choice
                    guided_count[choice] += 1
                particles.guide_region[indices] = selected_region
                allowed = (
                    self.body_plan.region[None]
                    == selected_region[:, None]
                )
                guide_distance = region_distance.gather(
                    1, selected_region[:, None]
                ).squeeze(1)
                far = guide_distance > (
                    self.config.hierarchical_arrival_distance_scale
                    * self.body_plan.spacing
                )
                recruitment_radius = torch.where(
                    far,
                    position.new_full(
                        (position.shape[0],),
                        self.config.hierarchical_site_radius_scale
                        * self.body_plan.spacing,
                    ),
                    position.new_full(
                        (position.shape[0],),
                        self.config.recruitment_radius_scale
                        * self.body_plan.spacing,
                    ),
                )[:, None]
            else:
                allowed = None
                recruitment_radius = (
                    self.config.recruitment_radius_scale
                    * self.body_plan.spacing
                )
            scores = (
                -site_distance.square() / (2 * recruitment_radius ** 2)
                + self.config.deficit_log_weight * torch.log(
                    deficit[None] + self.config.deficit_floor
                )
            )
            if allowed is not None:
                scores = scores.masked_fill(~allowed, -torch.inf)
            allocation = torch.softmax(scores, dim=1)
            field_target = allocation @ target_positions
            target_density = allocation @ self.body_plan.target_density
            entropy = -(
                allocation.clamp(min=1e-8).log() * allocation
            ).sum(dim=1) / math.log(self.body_plan.site_count)
        else:
            _, nearest_points, bone_distance = segment_projection(
                position, frame.endpoints
            )
            nearest_bone = bone_distance.argmin(dim=1)
            field_target = nearest_points[
                torch.arange(position.shape[0], device=position.device), nearest_bone
            ]
            if self.nearest_bone_uses_target_density:
                nearest_site = site_distance.argmin(dim=1)
                target_density = self.body_plan.target_density[nearest_site]
            else:
                target_density = position.new_full(
                    (position.shape[0],), self.body_plan.uniform_density
                )
            allocation = None
            entropy = position.new_zeros(position.shape[0])

        displacement = field_target - position
        attached = particles.material_locked[indices]
        if attached.any():
            attachment = particles.attachment_targets(frame, indices)
            committed_target = (
                (1 - self.config.moving_deficit_blend) * attachment
                + self.config.moving_deficit_blend * field_target
            )
            displacement = torch.where(
                attached[:, None], committed_target - position, displacement
            )
            attraction = torch.where(
                attached[:, None],
                position.new_full((position.shape[0], 1), self.config.attachment_attraction),
                position.new_full((position.shape[0], 1), self.config.attraction),
            )
        else:
            attraction = position.new_full((position.shape[0], 1), self.config.attraction)

        difference = position[:, None] - position[None, :]
        distance = difference.norm(dim=-1)
        identity = torch.eye(
            position.shape[0], device=position.device, dtype=torch.bool
        )
        direction = difference / distance.clamp(min=1e-6)[..., None]
        direction[identity] = 0
        density_radius = self.config.density_radius_scale * self.body_plan.spacing
        density_kernel = torch.exp(-(distance / density_radius).square())
        density_kernel[identity] = 0
        density = density_kernel.sum(dim=1)
        excess = torch.relu(density - target_density)
        pressure_range = torch.relu(1 - distance / density_radius).square()
        pressure_range[identity] = 0
        pair_pressure = (excess[:, None] + excess[None, :]) * pressure_range
        pressure_acceleration = (
            pair_pressure[..., None] * direction
        ).sum(dim=1) * self.config.pressure
        if not self.pressure_enabled:
            pressure_acceleration = torch.zeros_like(pressure_acceleration)

        minimum_distance = 0.62 * self.body_plan.spacing
        overlap = torch.relu(minimum_distance - distance) / minimum_distance
        overlap[identity] = 0
        repulsion = (
            overlap.square()[..., None] * direction
        ).sum(dim=1) * self.config.short_repulsion
        if not self.pressure_enabled:
            repulsion = torch.zeros_like(repulsion)

        acceleration = (
            attraction * displacement
            - self.config.damping * velocity
            + pressure_acceleration
            + repulsion
        )
        neighbor_offset = (
            density_kernel[..., None] * difference
        ).sum(dim=1) / density[:, None].clamp(min=1e-6)
        phase = torch.as_tensor(time, device=position.device, dtype=position.dtype)
        phase_features = torch.stack([torch.sin(phase), torch.cos(phase)]).expand(
            position.shape[0], -1
        )
        features = torch.cat([
            displacement,
            velocity,
            (density - target_density)[:, None],
            neighbor_offset,
            phase_features,
        ], dim=1)
        if learned_rule is not None:
            acceleration = acceleration + self.config.learned_acceleration_scale * learned_rule(
                features
            )

        next_velocity = velocity + self.config.dt * acceleration
        speed = next_velocity.norm(dim=1, keepdim=True)
        next_velocity = next_velocity * torch.clamp(
            self.config.maximum_speed / speed.clamp(min=1e-8), max=1
        )
        next_position = position + self.config.dt * next_velocity
        particles.velocities[indices] = next_velocity
        particles.positions[indices] = next_position
        particles.update_continuous_commitment(
            self.body_plan, frame, target_positions,
            self.config.commit_distance_scale * self.body_plan.spacing,
        )
        if not moving:
            particles.refresh_continuous_attachments(self.body_plan, frame)
            if self.plastic_material_enabled:
                particles.refresh_plastic_material(
                    self.body_plan, target_positions
                )
        matured = 0
        if allow_local_maturation:
            nearest_after = torch.cdist(
                next_position, target_positions
            ).amin(dim=1)
            density_relative_error = (
                (density - target_density).abs()
                / target_density.clamp(min=0.25)
            )
            stable = (
                (next_velocity.norm(dim=1) <= self.config.maturation_speed_max)
                & (
                    density_relative_error
                    <= self.config.maturation_density_tolerance
                )
                & (
                    nearest_after
                    <= self.config.maturation_distance_scale
                    * self.body_plan.spacing
                )
            )
            matured = particles.update_local_maturation(
                indices, stable, self.config.maturation_required_steps
            )

        return {
            "mean_site_deficit": float(deficit.mean().item()),
            "unfilled_site_fraction": float((deficit > 0.25).float().mean().item()),
            "allocation_entropy": float(entropy.mean().item()),
            "mean_density": float(density.mean().item()),
            "mean_pressure": float(excess.mean().item()),
            "locally_matured": matured,
        }
