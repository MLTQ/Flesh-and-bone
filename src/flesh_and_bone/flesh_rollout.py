"""H5 teacher-dataset training and autoregressive local-rule rollout."""

from dataclasses import dataclass

import torch

from .flesh_rule import FleshResidualRule, flesh_features
from .flesh_teacher import neighbor_mean_difference


@dataclass(frozen=True)
class FleshTrainingConfig:
    steps: int = 2400
    batch_size: int = 8192
    learning_rate: float = 2e-3
    weight_decay: float = 1e-6
    hidden_channels: int = 96
    rollout_cycles: int = 3


def teacher_dataset(trajectory, volume):
    """Flatten captured local states while retaining phase holdout labels."""
    phase_count, substeps, cell_count, _ = trajectory.residual.shape
    lbs_acceleration = trajectory.lbs_acceleration[:, None].expand(
        phase_count, substeps, cell_count, 3
    )
    bone_distance = volume.bone_distance[None, None, :, None].expand(
        phase_count, substeps, cell_count, 1
    )
    stiffness = trajectory.stiffness[None, None, :, None].expand_as(
        bone_distance
    )
    features = flesh_features(
        trajectory.residual,
        trajectory.velocity,
        lbs_acceleration,
        trajectory.neighbor_residual,
        trajectory.neighbor_velocity,
        bone_distance,
        stiffness,
    ).reshape(-1, 17)
    targets = trajectory.acceleration.reshape(-1, 3)
    phase = torch.arange(
        phase_count, device=features.device
    )[:, None, None].expand(
        phase_count, substeps, cell_count
    ).reshape(-1)
    return features, targets, phase


def train_flesh_rule(trajectory, volume, seed, config=None):
    """Train on non-holdout phases and report held-out normalized RMSE."""
    config = config or FleshTrainingConfig()
    torch.manual_seed(int(seed))
    features, targets, phase = teacher_dataset(trajectory, volume)
    holdout = phase.remainder(5) == 4
    train_indices = torch.nonzero(~holdout).flatten()
    holdout_indices = torch.nonzero(holdout).flatten()
    rule = FleshResidualRule(config.hidden_channels).to(features.device)
    rule.set_normalization(features[train_indices], targets[train_indices])
    optimizer = torch.optim.AdamW(
        rule.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    generator = torch.Generator(device=features.device)
    generator.manual_seed(int(seed) + 1009)
    rule.train()
    final_loss = 0.0
    for _ in range(config.steps):
        selected = torch.randint(
            train_indices.shape[0],
            (config.batch_size,),
            generator=generator,
            device=features.device,
        )
        batch = train_indices[selected]
        prediction = rule(features[batch])
        normalized_error = (
            prediction - targets[batch]
        ) / rule.target_std
        loss = normalized_error.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().item())
    rule.eval()
    with torch.no_grad():
        squared_error = 0.0
        squared_target = 0.0
        count = 0
        chunk = 65536
        target_mean = targets[holdout_indices].mean(dim=0)
        for start in range(0, holdout_indices.shape[0], chunk):
            indices = holdout_indices[start:start + chunk]
            prediction = rule(features[indices])
            squared_error += float(
                (prediction - targets[indices]).square().sum().item()
            )
            squared_target += float(
                (targets[indices] - target_mean).square().sum().item()
            )
            count += int(indices.shape[0])
    report = {
        "seed": int(seed),
        "steps": config.steps,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "train_examples": int(train_indices.shape[0]),
        "holdout_examples": int(holdout_indices.shape[0]),
        "final_normalized_loss": final_loss,
        "holdout_acceleration_nrmse": (
            squared_error / max(squared_target, 1e-12)
        ) ** 0.5,
        "holdout_acceleration_rmse": (
            squared_error / max(3 * count, 1)
        ) ** 0.5,
    }
    return rule, report


def rollout_flesh_rule(rule, trajectory, volume, graph, teacher_config,
                       cycles=3, neighbor_enabled=True):
    """Free-run one learned rule from the teacher's phase-zero state."""
    residual = trajectory.residual[0, 0].clone()
    velocity = trajectory.velocity[0, 0].clone()
    substep = 1.0 / (teacher_config.fps * teacher_config.substeps)
    cycles_out = []
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
                    if not neighbor_enabled:
                        neighbor_residual.zero_()
                        neighbor_velocity.zero_()
                    features = flesh_features(
                        residual,
                        velocity,
                        trajectory.lbs_acceleration[phase],
                        neighbor_residual,
                        neighbor_velocity,
                        volume.bone_distance[:, None],
                        trajectory.stiffness[:, None],
                    )
                    acceleration = rule(features)
                    velocity = velocity + substep * acceleration
                    residual = residual + substep * velocity
            cycles_out.append(torch.stack(phases))
    return torch.stack(cycles_out), residual, velocity
