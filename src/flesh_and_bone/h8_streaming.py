"""Cold-start, nonperiodic teacher and frozen-rule streaming for H8."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .density_rule import DensityStateDataset, current_density_scalar_features
from .density_teacher import density_observation, teacher_density_acceleration
from .flesh_rule import flesh_features
from .flesh_teacher import neighbor_mean_difference


@dataclass(frozen=True)
class H8MotionStream:
    """One nonperiodic LBS motion followed by an explicit final-pose hold."""

    lbs_positions: torch.Tensor
    bone_endpoints: torch.Tensor
    motion_frames: int
    hold_frames: int
    fps: float
    speed_multiplier: float

    @property
    def frame_count(self):
        return int(self.lbs_positions.shape[0])


@dataclass(frozen=True)
class H8TeacherTrajectory:
    """Visible streaming teacher states and a bounded diagnostic sample."""

    residual: torch.Tensor
    velocity: torch.Tensor
    lbs_positions: torch.Tensor
    lbs_acceleration: torch.Tensor
    stiffness: torch.Tensor
    final_residual: torch.Tensor
    final_velocity: torch.Tensor
    density_acceleration_max: float
    density_acceleration_rms: float
    density_cap_fraction: float
    finite: bool
    diagnostic_dataset: DensityStateDataset
    motion_frames: int
    hold_frames: int


@dataclass(frozen=True)
class H8StreamingRollout:
    """Visible cold-start rollout states and predicted-density safeguards."""

    residual: torch.Tensor
    velocity: torch.Tensor
    final_residual: torch.Tensor
    final_velocity: torch.Tensor
    density_acceleration_max: float
    density_acceleration_rms: float
    density_cap_fraction: float
    finite: bool


def nonperiodic_resample(values: torch.Tensor, speed_multiplier: float):
    """Linearly resample one finite sequence without wrapping its endpoints."""
    speed_multiplier = float(speed_multiplier)
    if speed_multiplier <= 0:
        raise ValueError("speed_multiplier must be positive")
    if values.shape[0] < 2:
        raise ValueError("nonperiodic resampling needs at least two frames")
    if abs(speed_multiplier - 1.0) <= 1e-12:
        return values.clone()
    output_intervals = max(1, round((values.shape[0] - 1) / speed_multiplier))
    time = torch.linspace(
        0,
        values.shape[0] - 1,
        output_intervals + 1,
        device=values.device,
        dtype=values.dtype,
    )
    lower = torch.floor(time).to(torch.long)
    upper = torch.minimum(lower + 1, lower.new_full((), values.shape[0] - 1))
    fraction = time - lower.to(time.dtype)
    shape = (fraction.shape[0],) + (1,) * (values.ndim - 1)
    return values[lower] * (1 - fraction.reshape(shape)) + values[upper] * fraction.reshape(shape)


def build_motion_stream(
    lbs_positions: torch.Tensor,
    bone_endpoints: torch.Tensor,
    speed_multiplier: float = 1.0,
    hold_frames: int = 30,
    fps: float = 30.0,
):
    """Resample matching LBS/bones and append an exact final-pose hold."""
    if lbs_positions.shape[0] != bone_endpoints.shape[0]:
        raise ValueError("LBS and bone frame counts differ")
    if int(hold_frames) < 1:
        raise ValueError("hold_frames must be positive")
    motion_lbs = nonperiodic_resample(lbs_positions, speed_multiplier)
    motion_bones = nonperiodic_resample(bone_endpoints, speed_multiplier)
    lbs = torch.cat([
        motion_lbs,
        motion_lbs[-1:].expand(int(hold_frames), *motion_lbs.shape[1:]),
    ])
    bones = torch.cat([
        motion_bones,
        motion_bones[-1:].expand(int(hold_frames), *motion_bones.shape[1:]),
    ])
    return H8MotionStream(
        lbs_positions=lbs,
        bone_endpoints=bones,
        motion_frames=int(motion_lbs.shape[0]),
        hold_frames=int(hold_frames),
        fps=float(fps),
        speed_multiplier=float(speed_multiplier),
    )


def nonperiodic_acceleration(positions: torch.Tensor, fps: float):
    """Second finite difference with non-wrapping one-sided endpoints."""
    if positions.shape[0] < 3:
        raise ValueError("acceleration needs at least three frames")
    acceleration = torch.empty_like(positions)
    acceleration[1:-1] = (
        positions[2:] - 2 * positions[1:-1] + positions[:-2]
    ) * float(fps) ** 2
    acceleration[0] = acceleration[1]
    acceleration[-1] = acceleration[-2]
    return acceleration


def _material_fields(volume, config):
    softness = (volume.bone_distance / config.softening_distance).clamp(0, 1)
    stiffness = (
        config.near_stiffness * (1 - softness)
        + config.far_stiffness * softness
    )
    damping = 2 * config.damping_ratio * torch.sqrt(stiffness)
    return softness, stiffness, damping


def simulate_streaming_teacher(
    stream: H8MotionStream,
    volume,
    graph,
    config,
    diagnostic_cells: int = 512,
    diagnostic_seed: int = 8801,
):
    """Simulate the explicit H7C teacher once from exactly zero dynamic state."""
    if stream.lbs_positions.shape[1:] != (volume.cell_count, 3):
        raise ValueError("stream and volume cell shapes differ")
    acceleration_lbs = nonperiodic_acceleration(
        stream.lbs_positions, stream.fps
    )
    softness, stiffness, damping = _material_fields(volume, config)
    residual = torch.zeros_like(volume.points)
    velocity = torch.zeros_like(residual)
    dt = 1.0 / (float(config.fps) * int(config.substeps))
    pitch = float(volume.metadata["pitch"])
    generator = torch.Generator(device=residual.device)
    generator.manual_seed(int(diagnostic_seed))
    selected = torch.randperm(
        volume.cell_count, generator=generator, device=residual.device
    )[:min(int(diagnostic_cells), volume.cell_count)]
    residual_frames = []
    velocity_frames = []
    scalar_samples = []
    compression_samples = []
    stretch_samples = []
    target_samples = []
    density_squared = residual.new_zeros(())
    density_vectors = 0
    density_max = residual.new_zeros(())
    near_cap = 0
    finite = True
    with torch.no_grad():
        for phase in range(stream.frame_count):
            residual_frames.append(residual.clone())
            velocity_frames.append(velocity.clone())
            for _ in range(int(config.substeps)):
                neighbor_residual = neighbor_mean_difference(residual, graph)
                neighbor_velocity = neighbor_mean_difference(velocity, graph)
                observation = density_observation(
                    stream.lbs_positions[phase], residual, graph, pitch, config
                )
                density_acceleration = teacher_density_acceleration(
                    observation, softness, config
                )
                scalars = current_density_scalar_features(
                    observation, velocity, volume, config
                )
                scalar_samples.append(scalars[selected])
                compression_samples.append(observation.compression_vector[selected])
                stretch_samples.append(observation.stretch_vector[selected])
                target_samples.append(density_acceleration[selected])
                acceleration = (
                    -stiffness[:, None] * residual
                    - damping[:, None] * velocity
                    + config.neighbor_coupling * neighbor_residual
                    - acceleration_lbs[phase]
                    + density_acceleration
                )
                norm = density_acceleration.norm(dim=-1)
                density_squared += density_acceleration.square().sum()
                density_vectors += density_acceleration.shape[0]
                density_max = torch.maximum(density_max, norm.max())
                near_cap += int((norm >= 0.99 * config.density_acceleration_cap).sum().item())
                velocity = velocity + dt * acceleration
                residual = residual + dt * velocity
                finite = finite and bool(
                    torch.isfinite(residual).all().item()
                    and torch.isfinite(velocity).all().item()
                )
    residual_out = torch.stack(residual_frames)
    velocity_out = torch.stack(velocity_frames)
    return H8TeacherTrajectory(
        residual=residual_out,
        velocity=velocity_out,
        lbs_positions=stream.lbs_positions,
        lbs_acceleration=acceleration_lbs,
        stiffness=stiffness,
        final_residual=residual,
        final_velocity=velocity,
        density_acceleration_max=float(density_max.item()),
        density_acceleration_rms=float(torch.sqrt(
            density_squared / max(density_vectors, 1)
        ).item()),
        density_cap_fraction=float(near_cap / max(density_vectors, 1)),
        finite=bool(finite and torch.isfinite(residual_out).all().item()),
        diagnostic_dataset=DensityStateDataset(
            scalars=torch.cat(scalar_samples),
            compression_vector=torch.cat(compression_samples),
            stretch_vector=torch.cat(stretch_samples),
            target=torch.cat(target_samples),
        ),
        motion_frames=stream.motion_frames,
        hold_frames=stream.hold_frames,
    )


def rollout_streaming_density(
    rule,
    trajectory: H8TeacherTrajectory,
    volume,
    graph,
    config,
    density_enabled: bool = True,
):
    """Run a frozen hybrid from zero state over one finite skeleton stream."""
    residual = torch.zeros_like(volume.points)
    velocity = torch.zeros_like(residual)
    dt = 1.0 / (float(config.fps) * int(config.substeps))
    pitch = float(volume.metadata["pitch"])
    residual_frames = []
    velocity_frames = []
    density_squared = residual.new_zeros(())
    density_vectors = 0
    density_max = residual.new_zeros(())
    near_cap = 0
    finite = True
    with torch.no_grad():
        for phase in range(trajectory.lbs_positions.shape[0]):
            residual_frames.append(residual.clone())
            velocity_frames.append(velocity.clone())
            for _ in range(int(config.substeps)):
                neighbor_residual = neighbor_mean_difference(residual, graph)
                neighbor_velocity = neighbor_mean_difference(velocity, graph)
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
                        config,
                    )
                    density_scalars = current_density_scalar_features(
                        observation, velocity, volume, config
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
                near_cap += int((norm >= 0.99 * config.density_acceleration_cap).sum().item())
                velocity = velocity + dt * acceleration
                residual = residual + dt * velocity
                finite = finite and bool(
                    torch.isfinite(residual).all().item()
                    and torch.isfinite(velocity).all().item()
                )
    residual_out = torch.stack(residual_frames)
    velocity_out = torch.stack(velocity_frames)
    return H8StreamingRollout(
        residual=residual_out,
        velocity=velocity_out,
        final_residual=residual,
        final_velocity=velocity,
        density_acceleration_max=float(density_max.item()),
        density_acceleration_rms=float(torch.sqrt(
            density_squared / max(density_vectors, 1)
        ).item()),
        density_cap_fraction=float(near_cap / max(density_vectors, 1)),
        finite=bool(finite and torch.isfinite(residual_out).all().item()),
    )
