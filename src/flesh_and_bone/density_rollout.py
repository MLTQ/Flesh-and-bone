"""Autoregressive rollout for the bounded H7 hybrid density rule."""

from dataclasses import dataclass

import torch

from .density_rule import current_density_scalar_features
from .density_teacher import density_observation
from .flesh_rule import flesh_features
from .flesh_teacher import neighbor_mean_difference


@dataclass(frozen=True)
class DensityRollout:
    """Repeated phase states plus safety diagnostics from one free run."""

    residual: torch.Tensor
    final_residual: torch.Tensor
    final_velocity: torch.Tensor
    density_acceleration_max: float
    density_acceleration_rms: float
    density_cap_fraction: float
    finite: bool


def rollout_hybrid_density(rule, trajectory, volume, graph, teacher_config,
                           cycles=20, density_enabled=True):
    """Free-run the hybrid from the nonlinear teacher's phase-zero state."""
    residual = trajectory.residual[0, 0].clone()
    velocity = trajectory.velocity[0, 0].clone()
    dt = 1.0 / (teacher_config.fps * teacher_config.substeps)
    pitch = float(volume.metadata["pitch"])
    cycles_out = []
    density_squared = residual.new_zeros(())
    density_vectors = 0
    density_max = residual.new_zeros(())
    near_cap = 0
    with torch.no_grad():
        for _ in range(int(cycles)):
            phases = []
            for phase in range(trajectory.lbs_positions.shape[0]):
                phases.append(residual.clone())
                for _ in range(teacher_config.substeps):
                    neighbor_residual = neighbor_mean_difference(
                        residual, graph
                    )
                    neighbor_velocity = neighbor_mean_difference(
                        velocity, graph
                    )
                    base_features = flesh_features(
                        residual,
                        velocity,
                        trajectory.lbs_acceleration[phase],
                        neighbor_residual,
                        neighbor_velocity,
                        volume.bone_distance[:, None],
                        trajectory.stiffness[:, None],
                    )
                    if density_enabled:
                        observation = density_observation(
                            trajectory.lbs_positions[phase],
                            residual,
                            graph,
                            pitch,
                            teacher_config,
                        )
                        density_scalars = current_density_scalar_features(
                            observation, velocity, volume, teacher_config
                        )
                        density_acceleration = rule.density_residual(
                            density_scalars,
                            observation.compression_vector,
                            observation.stretch_vector,
                        )
                    else:
                        density_acceleration = torch.zeros_like(residual)
                    acceleration = rule.backbone(base_features) + density_acceleration
                    norm = density_acceleration.norm(dim=-1)
                    density_squared += density_acceleration.square().sum()
                    density_vectors += density_acceleration.shape[0]
                    density_max = torch.maximum(density_max, norm.max())
                    near_cap += int((
                        norm >= 0.99 * teacher_config.density_acceleration_cap
                    ).sum().item())
                    velocity = velocity + dt * acceleration
                    residual = residual + dt * velocity
            cycles_out.append(torch.stack(phases))
    rollout = torch.stack(cycles_out)
    return DensityRollout(
        residual=rollout,
        final_residual=residual,
        final_velocity=velocity,
        density_acceleration_max=float(density_max.item()),
        density_acceleration_rms=float(torch.sqrt(
            density_squared / max(density_vectors, 1)
        ).item()),
        density_cap_fraction=float(
            near_cap / max(density_vectors, 1)
        ),
        finite=bool(
            torch.isfinite(rollout).all().item()
            and torch.isfinite(residual).all().item()
            and torch.isfinite(velocity).all().item()
        ),
    )
