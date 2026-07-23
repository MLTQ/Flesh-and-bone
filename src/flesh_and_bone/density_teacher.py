"""Nonlinear edge-density teacher layered on graph-elastic flesh mechanics."""

from dataclasses import dataclass

import torch

from .flesh_teacher import (
    ElasticTeacherConfig,
    TeacherTrajectory,
    neighbor_mean_difference,
    periodic_acceleration,
    volume_lbs_cycle,
)


@dataclass(frozen=True)
class DensityTeacherConfig:
    """Frozen H7 nonlinear-mechanics constants and integration settings."""

    fps: float = 30.0
    substeps: int = 4
    warmup_cycles: int = 30
    near_stiffness: float = 1200.0
    far_stiffness: float = 300.0
    softening_distance: float = 0.18
    damping_ratio: float = 0.22
    neighbor_coupling: float = 1200.0
    compression_threshold: float = 0.05
    stretch_threshold: float = 0.08
    strain_clip: float = 1.0
    denominator_pitch_fraction: float = 0.5
    pressure_near: float = 30.0
    pressure_far: float = 50.0
    cohesion_near: float = 6.0
    cohesion_far: float = 12.0
    density_acceleration_cap: float = 12.0

    def elastic_config(self):
        """Return the exactly corresponding linear-backbone configuration."""
        return ElasticTeacherConfig(
            fps=self.fps,
            substeps=self.substeps,
            warmup_cycles=self.warmup_cycles,
            near_stiffness=self.near_stiffness,
            far_stiffness=self.far_stiffness,
            softening_distance=self.softening_distance,
            damping_ratio=self.damping_ratio,
            neighbor_coupling=self.neighbor_coupling,
        )


@dataclass(frozen=True)
class DensityObservation:
    """Invariant scalars and equivariant vectors derived from local edges."""

    signed_compression: torch.Tensor
    compression_rms: torch.Tensor
    stretch_rms: torch.Tensor
    compression_vector: torch.Tensor
    stretch_vector: torch.Tensor


@dataclass(frozen=True)
class DensityTeacherTrajectory(TeacherTrajectory):
    """Captured nonlinear teacher states plus density supervision."""

    signed_compression: torch.Tensor
    compression_rms: torch.Tensor
    stretch_rms: torch.Tensor
    compression_vector: torch.Tensor
    stretch_vector: torch.Tensor
    density_acceleration: torch.Tensor


def smooth_norm_cap(vector, maximum):
    """Smoothly bound vector norm while preserving its direction exactly."""
    maximum = float(maximum)
    if maximum <= 0:
        raise ValueError("maximum must be positive")
    norm = vector.norm(dim=-1, keepdim=True)
    scale = maximum * torch.tanh(norm / maximum) / norm.clamp(min=1e-12)
    return vector * scale


def density_observation(lbs_positions, residual, graph, pitch, config=None):
    """Measure directed axial excess strain relative to the current LBS pose."""
    config = config or DensityTeacherConfig()
    equilibrium = (
        lbs_positions[graph.target] - lbs_positions[graph.source]
    )
    equilibrium_length = equilibrium.norm(dim=-1)
    unit = equilibrium / equilibrium_length[:, None].clamp(min=1e-12)
    denominator = equilibrium_length.clamp(
        min=config.denominator_pitch_fraction * float(pitch)
    )
    difference = residual[graph.target] - residual[graph.source]
    strain = (difference * unit).sum(dim=-1) / denominator
    strain = strain.clamp(-config.strain_clip, config.strain_clip)
    compression = torch.relu(-strain - config.compression_threshold)
    stretch = torch.relu(strain - config.stretch_threshold)

    cells = residual.shape[0]
    degree = graph.degree.clamp(min=1)

    def mean_scalar(value):
        result = residual.new_zeros(cells)
        result.index_add_(0, graph.source, value)
        return result / degree

    def mean_vector(value):
        result = torch.zeros_like(residual)
        result.index_add_(0, graph.source, value)
        return result / degree[:, None]

    return DensityObservation(
        signed_compression=mean_scalar(-strain),
        compression_rms=torch.sqrt(mean_scalar(compression.square())),
        stretch_rms=torch.sqrt(mean_scalar(stretch.square())),
        compression_vector=mean_vector(-compression.square()[:, None] * unit),
        stretch_vector=mean_vector(stretch.square()[:, None] * unit),
    )


def teacher_density_acceleration(observation, softness, config=None):
    """Apply the frozen spatially varying pressure and cohesion coefficients."""
    config = config or DensityTeacherConfig()
    pressure = config.pressure_near * (1 - softness) + config.pressure_far * softness
    cohesion = config.cohesion_near * (1 - softness) + config.cohesion_far * softness
    raw = (
        pressure[:, None] * observation.compression_vector
        + cohesion[:, None] * observation.stretch_vector
    )
    return smooth_norm_cap(raw, config.density_acceleration_cap)


def simulate_density_teacher_from_lbs(
    lbs_positions, volume, graph, config=None
):
    """Converge and capture the H7 teacher around a periodic LBS cycle."""
    config = config or DensityTeacherConfig()
    if lbs_positions.shape[1:] != (volume.cell_count, 3):
        raise ValueError("LBS cycle and volume cell shapes differ")
    lbs_acceleration = periodic_acceleration(lbs_positions, config.fps)
    softness = (volume.bone_distance / config.softening_distance).clamp(0, 1)
    stiffness = (
        config.near_stiffness * (1 - softness)
        + config.far_stiffness * softness
    )
    damping = 2 * config.damping_ratio * torch.sqrt(stiffness)
    residual = torch.zeros_like(volume.points)
    velocity = torch.zeros_like(residual)
    dt = 1.0 / (config.fps * config.substeps)
    pitch = float(volume.metadata["pitch"])

    captured_names = (
        "residual", "velocity", "acceleration", "neighbor_residual",
        "neighbor_velocity", "signed_compression", "compression_rms",
        "stretch_rms", "compression_vector", "stretch_vector",
        "density_acceleration",
    )

    def advance(phase, capture=None):
        nonlocal residual, velocity
        for _ in range(config.substeps):
            neighbor_residual = neighbor_mean_difference(residual, graph)
            neighbor_velocity = neighbor_mean_difference(velocity, graph)
            observation = density_observation(
                lbs_positions[phase], residual, graph, pitch, config
            )
            density_acceleration = teacher_density_acceleration(
                observation, softness, config
            )
            acceleration = (
                -stiffness[:, None] * residual
                - damping[:, None] * velocity
                + config.neighbor_coupling * neighbor_residual
                - lbs_acceleration[phase]
                + density_acceleration
            )
            if capture is not None:
                values = {
                    "residual": residual,
                    "velocity": velocity,
                    "acceleration": acceleration,
                    "neighbor_residual": neighbor_residual,
                    "neighbor_velocity": neighbor_velocity,
                    "signed_compression": observation.signed_compression,
                    "compression_rms": observation.compression_rms,
                    "stretch_rms": observation.stretch_rms,
                    "compression_vector": observation.compression_vector,
                    "stretch_vector": observation.stretch_vector,
                    "density_acceleration": density_acceleration,
                }
                for name, value in values.items():
                    capture[name].append(value.clone())
            velocity = velocity + dt * acceleration
            residual = residual + dt * velocity

    with torch.no_grad():
        for _ in range(config.warmup_cycles):
            for phase in range(lbs_positions.shape[0]):
                advance(phase)
        captured = {name: [] for name in captured_names}
        for phase in range(lbs_positions.shape[0]):
            advance(phase, captured)

    vector_shape = (
        lbs_positions.shape[0], config.substeps, volume.cell_count, 3
    )
    scalar_shape = vector_shape[:-1]

    def stacked(name, shape):
        return torch.stack(captured[name]).reshape(shape)

    return DensityTeacherTrajectory(
        residual=stacked("residual", vector_shape),
        velocity=stacked("velocity", vector_shape),
        acceleration=stacked("acceleration", vector_shape),
        neighbor_residual=stacked("neighbor_residual", vector_shape),
        neighbor_velocity=stacked("neighbor_velocity", vector_shape),
        lbs_positions=lbs_positions,
        lbs_acceleration=lbs_acceleration,
        stiffness=stiffness,
        final_residual=residual,
        final_velocity=velocity,
        signed_compression=stacked("signed_compression", scalar_shape),
        compression_rms=stacked("compression_rms", scalar_shape),
        stretch_rms=stacked("stretch_rms", scalar_shape),
        compression_vector=stacked("compression_vector", vector_shape),
        stretch_vector=stacked("stretch_vector", vector_shape),
        density_acceleration=stacked("density_acceleration", vector_shape),
    )


def simulate_density_teacher(asset, volume, graph, config=None):
    """Converge the nonlinear teacher around the rig asset's walk cycle."""
    config = config or DensityTeacherConfig()
    lbs_positions, _ = volume_lbs_cycle(asset, volume)
    return simulate_density_teacher_from_lbs(
        lbs_positions, volume, graph, config=config
    )
