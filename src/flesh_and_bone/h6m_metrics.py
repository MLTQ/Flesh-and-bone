"""Frozen-feature shift and strict H6M acceptance measurements."""

import torch

from .flesh_rule import flesh_features


FEATURE_NAMES = (
    "residual_x", "residual_y", "residual_z",
    "velocity_x", "velocity_y", "velocity_z",
    "lbs_acceleration_x", "lbs_acceleration_y", "lbs_acceleration_z",
    "neighbor_residual_x", "neighbor_residual_y", "neighbor_residual_z",
    "neighbor_velocity_x", "neighbor_velocity_y", "neighbor_velocity_z",
    "bone_distance", "stiffness",
)


def measure_frozen_feature_shift(trajectory, volume, rule,
                                 sample_examples=131072):
    """Sample a trajectory and measure z-shift under checkpoint normalization."""
    phases, substeps, cells, _ = trajectory.residual.shape
    total = phases * substeps * cells
    count = min(int(sample_examples), total)
    indices = torch.linspace(
        0,
        total - 1,
        count,
        device=trajectory.residual.device,
        dtype=torch.float64,
    ).round().to(torch.long).unique()
    phase = torch.div(indices, substeps * cells, rounding_mode="floor")
    cell = indices.remainder(cells)

    def sampled(value):
        return value.reshape(-1, 3)[indices]

    features = flesh_features(
        sampled(trajectory.residual),
        sampled(trajectory.velocity),
        trajectory.lbs_acceleration[phase, cell],
        sampled(trajectory.neighbor_residual),
        sampled(trajectory.neighbor_velocity),
        volume.bone_distance[cell, None],
        trajectory.stiffness[cell, None],
    )
    z = (features - rule.feature_mean) / rule.feature_std
    absolute = z.abs()
    acceleration = absolute[:, 6:9]
    per_channel = torch.sqrt(z.square().mean(dim=0))
    return {
        "sample_examples": int(indices.shape[0]),
        "z_rms": float(torch.sqrt(z.square().mean()).item()),
        "z_abs_p99": float(torch.quantile(absolute, 0.99).item()),
        "z_abs_max": float(absolute.max().item()),
        "fraction_abs_z_over_3": float((absolute > 3).float().mean().item()),
        "acceleration_z_abs_p99": float(
            torch.quantile(acceleration, 0.99).item()
        ),
        "acceleration_z_abs_max": float(acceleration.max().item()),
        "per_channel_z_rms": {
            name: float(value)
            for name, value in zip(FEATURE_NAMES, per_channel.tolist())
        },
    }


def acceptance_h6m(teacher, learned, neighbor_blind):
    """Apply the unchanged H5 rollout gates plus H6M teacher bounds."""
    neighbor_causal = (
        neighbor_blind["position_rms"] >= 1.20 * learned["position_rms"]
        or neighbor_blind["edge_strain_error_rms"]
        >= 1.25 * learned["edge_strain_error_rms"]
    )
    gates = {
        "teacher_finite": teacher["finite"],
        "teacher_periodic": teacher["cycle_seam_rms"] <= 5e-4,
        "teacher_bounded": 0.001 <= teacher["residual_rms"] <= 0.050,
        "teacher_coherent": teacher["edge_difference_rms"] < 0.025,
        "rollout_rms": learned["position_rms"] <= 0.004,
        "rollout_p99": learned["position_p99"] <= 0.012,
        "rollout_max": learned["position_max"] <= 0.040,
        "rollout_amplitude": 0.75 <= learned["amplitude_ratio"] <= 1.25,
        "rollout_softness": learned["far_near_amplitude_ratio"] >= 1.25,
        "rollout_drift": learned["phase_zero_cycle_drift_rms"] <= 0.003,
        "rollout_finite": learned["finite"],
        "beats_lbs": learned["lbs_improvement_fraction"] >= 0.35,
        "neighbor_causal": neighbor_causal,
    }
    return {**gates, "pass": all(gates.values())}
