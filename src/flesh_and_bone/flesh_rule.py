"""Shared normalized local MLP for H5 residual acceleration."""

import torch
from torch import nn


FEATURE_CHANNELS = 17


def flesh_features(residual, velocity, lbs_acceleration, neighbor_residual,
                   neighbor_velocity, bone_distance, stiffness):
    """Assemble raw per-cell local teacher/rollout features."""
    prefix = residual.shape[:-1]
    return torch.cat([
        residual,
        velocity,
        lbs_acceleration.expand(*prefix, 3),
        neighbor_residual,
        neighbor_velocity,
        bone_distance.expand(*prefix, 1),
        stiffness.expand(*prefix, 1),
    ], dim=-1)


class FleshResidualRule(nn.Module):
    """Predict raw residual acceleration from normalized local messages."""

    def __init__(self, hidden_channels=96):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(FEATURE_CHANNELS, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 3),
        )
        self.register_buffer("feature_mean", torch.zeros(FEATURE_CHANNELS))
        self.register_buffer("feature_std", torch.ones(FEATURE_CHANNELS))
        self.register_buffer("target_mean", torch.zeros(3))
        self.register_buffer("target_std", torch.ones(3))

    def set_normalization(self, features, targets):
        """Fit fixed dataset normalization without making it trainable state."""
        with torch.no_grad():
            self.feature_mean.copy_(features.mean(dim=0))
            self.feature_std.copy_(features.std(dim=0).clamp(min=1e-6))
            self.target_mean.copy_(targets.mean(dim=0))
            self.target_std.copy_(targets.std(dim=0).clamp(min=1e-6))

    def forward(self, features):
        normalized = (features - self.feature_mean) / self.feature_std
        output = self.network(normalized)
        return output * self.target_std + self.target_mean
