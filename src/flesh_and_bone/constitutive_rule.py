"""Structure-preserving five-term local constitutive rule and identifier."""

from dataclasses import dataclass

import torch
from torch import nn

from .flesh_rule import flesh_features


COEFFICIENT_NAMES = (
    "stiffness",
    "damping",
    "neighbor_residual",
    "neighbor_velocity",
    "lbs_inertia",
)


def constitutive_terms(features):
    """Convert H5 raw features into five vector-valued physical basis terms."""
    stiffness = features[..., 16:17].clamp(min=0)
    return torch.stack([
        -stiffness * features[..., 0:3],
        -torch.sqrt(stiffness) * features[..., 3:6],
        features[..., 9:12],
        features[..., 12:15],
        -features[..., 6:9],
    ], dim=-2)


class ConstitutiveFleshRule(nn.Module):
    """Apply shared fitted scalars to stable equivariant physical terms."""

    def __init__(self, coefficients):
        super().__init__()
        coefficients = torch.as_tensor(coefficients)
        if coefficients.shape != (5,):
            raise ValueError("constitutive rule needs five coefficients")
        self.register_buffer("coefficients", coefficients)

    def forward(self, features):
        terms = constitutive_terms(features)
        return torch.einsum("...kc,k->...c", terms, self.coefficients)


@dataclass(frozen=True)
class ConstitutiveFit:
    coefficients: tuple[float, ...]
    train_state_vectors: int
    holdout_state_vectors: int
    holdout_acceleration_nrmse: float
    holdout_acceleration_rmse: float


def _trajectory_features(trajectory, volume, phase, substep):
    cells = trajectory.residual.shape[2]
    return flesh_features(
        trajectory.residual[phase, substep],
        trajectory.velocity[phase, substep],
        trajectory.lbs_acceleration[phase],
        trajectory.neighbor_residual[phase, substep],
        trajectory.neighbor_velocity[phase, substep],
        volume.bone_distance[:, None],
        trajectory.stiffness[:, None],
    ).reshape(cells, 17)


def fit_constitutive_rule(trajectory, volume):
    """Identify five shared coefficients from non-holdout walk phases."""
    phases, substeps, cells, _ = trajectory.residual.shape
    device = trajectory.residual.device
    normal = torch.zeros(5, 5, device=device, dtype=torch.float64)
    rhs = torch.zeros(5, device=device, dtype=torch.float64)
    train_states = 0
    for phase in range(phases):
        if phase % 5 == 4:
            continue
        for substep in range(substeps):
            features = _trajectory_features(
                trajectory, volume, phase, substep
            )
            design = constitutive_terms(features).to(torch.float64)
            design = design.transpose(1, 2).reshape(-1, 5)
            target = trajectory.acceleration[
                phase, substep
            ].to(torch.float64).reshape(-1)
            normal += design.T @ design
            rhs += design.T @ target
            train_states += cells

    scale = torch.sqrt(torch.diagonal(normal)).clamp(min=1e-12)
    normalized = normal / (scale[:, None] * scale[None, :])
    normalized_rhs = rhs / scale
    solved = torch.linalg.solve(normalized, normalized_rhs) / scale

    holdout_targets = trajectory.acceleration[
        torch.arange(phases, device=device).remainder(5) == 4
    ].to(torch.float64)
    target_mean = holdout_targets.mean(dim=(0, 1, 2))
    squared_error = torch.zeros((), device=device, dtype=torch.float64)
    squared_target = torch.zeros((), device=device, dtype=torch.float64)
    holdout_states = 0
    for phase in range(phases):
        if phase % 5 != 4:
            continue
        for substep in range(substeps):
            features = _trajectory_features(
                trajectory, volume, phase, substep
            )
            terms = constitutive_terms(features).to(torch.float64)
            prediction = torch.einsum("nkj,k->nj", terms, solved)
            target = trajectory.acceleration[phase, substep].to(torch.float64)
            squared_error += (prediction - target).square().sum()
            squared_target += (target - target_mean).square().sum()
            holdout_states += cells
    report = ConstitutiveFit(
        coefficients=tuple(float(value) for value in solved.tolist()),
        train_state_vectors=int(train_states),
        holdout_state_vectors=int(holdout_states),
        holdout_acceleration_nrmse=float(
            torch.sqrt(squared_error / squared_target.clamp(min=1e-24)).item()
        ),
        holdout_acceleration_rmse=float(
            torch.sqrt(
                squared_error / max(3 * holdout_states, 1)
            ).item()
        ),
    )
    rule = ConstitutiveFleshRule(
        solved.to(device=device, dtype=trajectory.residual.dtype)
    )
    return rule, report
