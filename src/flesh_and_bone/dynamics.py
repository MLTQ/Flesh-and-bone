"""Density-regulated mechanics and learned particle-NCA residual interface."""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class DynamicsConfig:
    dt: float = 0.035
    attraction: float = 18.0
    damping: float = 4.5
    pressure: float = 6.0
    short_repulsion: float = 12.0
    density_radius_scale: float = 1.9
    commit_distance_scale: float = 0.72
    maximum_speed: float = 2.4
    learned_acceleration_scale: float = 0.25


class ParticleNCARule(nn.Module):
    """Shared local residual rule reserved for H1 training."""

    input_channels = 12

    def __init__(self, hidden=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(self.input_channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features):
        return self.network(features)


class MechanicalDynamics:
    """H0 upper-bound controller with density pressure and optional NCA residual."""

    def __init__(self, body_plan, config=None):
        self.body_plan = body_plan
        self.config = config or DynamicsConfig()

    def step(self, particles, target_positions, time, learned_rule=None):
        indices, position, velocity, assignments = particles.active_tensors()
        if not indices.numel():
            return {"mean_density": 0.0, "mean_pressure": 0.0}
        target = target_positions[assignments]
        displacement = target - position

        difference = position[:, None] - position[None, :]
        distance = difference.norm(dim=-1)
        identity = torch.eye(
            position.shape[0], device=position.device, dtype=torch.bool
        )
        safe_distance = distance.clamp(min=1e-6)
        direction = difference / safe_distance[..., None]
        direction[identity] = 0

        density_radius = self.config.density_radius_scale * self.body_plan.spacing
        density_kernel = torch.exp(-(distance / density_radius).square())
        density_kernel[identity] = 0
        density = density_kernel.sum(dim=1)
        target_density = self.body_plan.target_density[assignments]
        excess = torch.relu(density - target_density)

        pressure_range = torch.relu(1 - distance / density_radius).square()
        pressure_range[identity] = 0
        pair_pressure = (excess[:, None] + excess[None, :]) * pressure_range
        pressure_acceleration = (
            pair_pressure[..., None] * direction
        ).sum(dim=1) * self.config.pressure

        minimum_distance = 0.62 * self.body_plan.spacing
        overlap = torch.relu(minimum_distance - distance) / minimum_distance
        overlap[identity] = 0
        repulsion = (
            overlap.square()[..., None] * direction
        ).sum(dim=1) * self.config.short_repulsion

        acceleration = (
            self.config.attraction * displacement
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
        particles.update_commitment(
            self.body_plan, target_positions,
            self.config.commit_distance_scale * self.body_plan.spacing,
        )
        return {
            "mean_density": float(density.mean().item()),
            "mean_pressure": float(excess.mean().item()),
        }
