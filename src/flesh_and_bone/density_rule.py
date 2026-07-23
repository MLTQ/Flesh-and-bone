"""Bounded invariant coefficient learner for H7 density mechanics."""

from dataclasses import dataclass

import torch
from torch import nn

from .density_teacher import smooth_norm_cap


@dataclass(frozen=True)
class DensityTrainingConfig:
    """Frozen initial H7 optimizer and capacity settings."""

    steps: int = 1200
    batch_size: int = 65536
    learning_rate: float = 2e-3
    weight_decay: float = 1e-6
    hidden_channels: int = 32
    examples_per_motion: int = 300000
    pressure_max: float = 60.0
    cohesion_max: float = 18.0
    density_acceleration_cap: float = 12.0


@dataclass(frozen=True)
class DensityStateDataset:
    """Deterministically sampled invariant states and vector targets."""

    scalars: torch.Tensor
    compression_vector: torch.Tensor
    stretch_vector: torch.Tensor
    target: torch.Tensor

    @property
    def example_count(self):
        return self.scalars.shape[0]


class BoundedDensityResidual(nn.Module):
    """Predict bounded nonnegative coefficients for two explicit mechanisms."""

    def __init__(self, hidden_channels=32, pressure_max=60.0,
                 cohesion_max=18.0, acceleration_cap=12.0):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(5, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 2),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.register_buffer(
            "coefficient_maxima",
            torch.tensor([pressure_max, cohesion_max], dtype=torch.float32),
        )
        self.acceleration_cap = float(acceleration_cap)

    def coefficients(self, scalars):
        """Return pressure/cohesion coefficients inside declared bounds."""
        return torch.sigmoid(self.network(scalars)) * self.coefficient_maxima

    def forward(self, scalars, compression_vector, stretch_vector):
        coefficients = self.coefficients(scalars)
        acceleration = (
            coefficients[..., 0:1] * compression_vector
            + coefficients[..., 1:2] * stretch_vector
        )
        return smooth_norm_cap(acceleration, self.acceleration_cap)


class HybridDensityRule(nn.Module):
    """Frozen constitutive backbone plus one bounded nonlinear residual."""

    def __init__(self, backbone, density_residual):
        super().__init__()
        self.backbone = backbone
        self.density_residual = density_residual
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

    def forward(self, base_features, density_scalars, compression_vector,
                stretch_vector, density_enabled=True):
        acceleration = self.backbone(base_features)
        if density_enabled:
            acceleration = acceleration + self.density_residual(
                density_scalars, compression_vector, stretch_vector
            )
        return acceleration


def current_density_scalar_features(observation, velocity, volume, config):
    """Build the five invariant inputs for a live autoregressive state."""
    pitch = float(volume.metadata["pitch"])
    softness = (
        volume.bone_distance / float(config.softening_distance)
    ).clamp(0, 1)
    speed = velocity.norm(dim=-1) / (pitch * float(config.fps))
    return torch.stack([
        observation.signed_compression.clamp(-1, 1),
        observation.compression_rms.clamp(0, 1),
        observation.stretch_rms.clamp(0, 1),
        softness,
        speed.clamp(0, 4),
    ], dim=-1)


def density_scalar_features(trajectory, volume, flat_indices, config):
    """Gather the five declared bounded invariant inputs from captured states."""
    cells = trajectory.residual.shape[2]
    cell = flat_indices.remainder(cells)
    velocity = trajectory.velocity.reshape(-1, 3)[flat_indices]
    pitch = float(volume.metadata["pitch"])
    speed = velocity.norm(dim=-1) / (pitch * float(config.fps))
    softness = (
        volume.bone_distance[cell] / float(config.softening_distance)
    ).clamp(0, 1)
    return torch.stack([
        trajectory.signed_compression.reshape(-1)[flat_indices].clamp(-1, 1),
        trajectory.compression_rms.reshape(-1)[flat_indices].clamp(0, 1),
        trajectory.stretch_rms.reshape(-1)[flat_indices].clamp(0, 1),
        softness,
        speed.clamp(0, 4),
    ], dim=-1)


def sample_density_states(trajectory, volume, count, seed, teacher_config):
    """Sample captured states without materializing a full feature matrix."""
    device = trajectory.residual.device
    total = trajectory.residual[..., 0].numel()
    count = min(int(count), total)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    indices = torch.randint(
        total, (count,), generator=generator, device=device
    )
    return DensityStateDataset(
        scalars=density_scalar_features(
            trajectory, volume, indices, teacher_config
        ),
        compression_vector=trajectory.compression_vector.reshape(
            -1, 3
        )[indices],
        stretch_vector=trajectory.stretch_vector.reshape(-1, 3)[indices],
        target=trajectory.density_acceleration.reshape(-1, 3)[indices],
    )


def concatenate_density_datasets(datasets):
    """Join motion-balanced samples after each large trajectory is released."""
    datasets = tuple(datasets)
    if not datasets:
        raise ValueError("at least one density dataset is required")
    return DensityStateDataset(*(
        torch.cat([getattr(item, field) for item in datasets], dim=0)
        for field in (
            "scalars", "compression_vector", "stretch_vector", "target"
        )
    ))


def density_acceleration_nrmse(rule, dataset, chunk_size=131072):
    """Measure vector RMSE normalized by centered target energy."""
    target_mean = dataset.target.mean(dim=0)
    squared_error = dataset.target.new_zeros(())
    squared_target = dataset.target.new_zeros(())
    maximum = dataset.target.new_zeros(())
    with torch.no_grad():
        for start in range(0, dataset.example_count, int(chunk_size)):
            stop = start + int(chunk_size)
            prediction = rule(
                dataset.scalars[start:stop],
                dataset.compression_vector[start:stop],
                dataset.stretch_vector[start:stop],
            )
            target = dataset.target[start:stop]
            squared_error += (prediction - target).square().sum()
            squared_target += (target - target_mean).square().sum()
            maximum = torch.maximum(maximum, prediction.norm(dim=-1).max())
    return {
        "acceleration_nrmse": float(torch.sqrt(
            squared_error / squared_target.clamp(min=1e-24)
        ).item()),
        "acceleration_rmse": float(torch.sqrt(
            squared_error / max(3 * dataset.example_count, 1)
        ).item()),
        "predicted_acceleration_max": float(maximum.item()),
        "examples": int(dataset.example_count),
    }


def train_density_residual(dataset, seed, config=None):
    """Fit only the bounded density coefficients on motion-balanced samples."""
    config = config or DensityTrainingConfig()
    torch.manual_seed(int(seed))
    rule = BoundedDensityResidual(
        hidden_channels=config.hidden_channels,
        pressure_max=config.pressure_max,
        cohesion_max=config.cohesion_max,
        acceleration_cap=config.density_acceleration_cap,
    ).to(dataset.scalars.device)
    optimizer = torch.optim.AdamW(
        rule.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator(device=dataset.scalars.device)
    generator.manual_seed(int(seed) + 1709)
    target_scale = torch.sqrt(dataset.target.square().mean()).clamp(min=1e-8)
    final_loss = 0.0
    rule.train()
    for _ in range(config.steps):
        selected = torch.randint(
            dataset.example_count,
            (config.batch_size,),
            generator=generator,
            device=dataset.scalars.device,
        )
        prediction = rule(
            dataset.scalars[selected],
            dataset.compression_vector[selected],
            dataset.stretch_vector[selected],
        )
        error = (prediction - dataset.target[selected]) / target_scale
        loss = error.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().item())
    rule.eval()
    report = density_acceleration_nrmse(rule, dataset)
    report.update({
        "seed": int(seed),
        "steps": config.steps,
        "batch_size": config.batch_size,
        "hidden_channels": config.hidden_channels,
        "final_normalized_loss": final_loss,
    })
    return rule, report
