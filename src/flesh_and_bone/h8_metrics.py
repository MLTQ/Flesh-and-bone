"""Streaming teacher, rollout, settling, and acceptance metrics for H8."""

import torch

from .h5_metrics import flat_quantile


def _vector_rms(values):
    return torch.sqrt(values.square().sum(dim=-1).mean())


def measure_h8_teacher(trajectory, volume, graph):
    """Measure cold-start teacher amplitude, locality, density, and settling."""
    residual = trajectory.residual
    amplitude = residual.norm(dim=-1)
    far = volume.bone_distance > 0.12
    near = volume.bone_distance < 0.05
    edge = residual[:, graph.target] - residual[:, graph.source]
    entry_velocity = _vector_rms(trajectory.velocity[trajectory.motion_frames])
    final_velocity = _vector_rms(trajectory.final_velocity)
    return {
        "finite": bool(trajectory.finite),
        "frame_count": int(residual.shape[0]),
        "motion_frames": int(trajectory.motion_frames),
        "hold_frames": int(trajectory.hold_frames),
        "residual_rms": float(_vector_rms(residual).item()),
        "final_residual_rms": float(_vector_rms(trajectory.final_residual).item()),
        "far_near_amplitude_ratio": float(
            (amplitude[:, far].mean() / amplitude[:, near].mean().clamp(min=1e-12)).item()
        ),
        "edge_difference_rms": float(_vector_rms(edge).item()),
        "hold_entry_velocity_rms": float(entry_velocity.item()),
        "final_velocity_rms": float(final_velocity.item()),
        "hold_velocity_decay_ratio": float(
            (final_velocity / entry_velocity.clamp(min=1e-12)).item()
        ),
        "density_acceleration_rms": trajectory.density_acceleration_rms,
        "density_acceleration_max": trajectory.density_acceleration_max,
        "density_cap_fraction": trajectory.density_cap_fraction,
    }


def _compression_excess(lbs, residual, source, target, pitch, config):
    equilibrium = lbs[target] - lbs[source]
    length = equilibrium.norm(dim=-1)
    unit = equilibrium / length[:, None].clamp(min=1e-12)
    denominator = length.clamp(
        min=config.denominator_pitch_fraction * float(pitch)
    )
    difference = residual[target] - residual[source]
    strain = ((difference * unit).sum(dim=-1) / denominator).clamp(
        -config.strain_clip, config.strain_clip
    )
    return torch.relu(-strain - config.compression_threshold)


def measure_h8_compression_error(
    rollout, trajectory, graph, pitch, config, edge_stride=8
):
    """Compare sampled compression over one finite stream without cycle axes."""
    source = graph.source[::int(edge_stride)]
    target = graph.target[::int(edge_stride)]
    squared_error = rollout.new_zeros(())
    squared_target = rollout.new_zeros(())
    count = 0
    for phase in range(rollout.shape[0]):
        teacher_value = _compression_excess(
            trajectory.lbs_positions[phase],
            trajectory.residual[phase],
            source,
            target,
            pitch,
            config,
        )
        prediction = _compression_excess(
            trajectory.lbs_positions[phase],
            rollout[phase],
            source,
            target,
            pitch,
            config,
        )
        squared_error += (prediction - teacher_value).square().sum()
        squared_target += teacher_value.square().sum()
        count += teacher_value.numel()
    return {
        "compression_error_rms": float(torch.sqrt(
            squared_error / max(count, 1)
        ).item()),
        "compression_target_rms": float(torch.sqrt(
            squared_target / max(count, 1)
        ).item()),
        "sampled_directed_edges": int(source.numel()),
    }


def measure_h8_rollout(result, trajectory, volume, graph, config):
    """Compare one streaming rollout to the cold-start explicit teacher."""
    teacher = trajectory.residual
    error = result.residual - teacher
    distance = error.norm(dim=-1)
    amplitude = result.residual.norm(dim=-1)
    far = volume.bone_distance > 0.12
    near = volume.bone_distance < 0.05
    teacher_rms = _vector_rms(teacher)
    rollout_rms = _vector_rms(result.residual)
    entry_velocity = _vector_rms(result.velocity[trajectory.motion_frames])
    final_velocity = _vector_rms(result.final_velocity)
    report = {
        "position_rms": float(_vector_rms(error).item()),
        "position_p99": float(flat_quantile(distance, 0.99).item()),
        "position_max": float(distance.max().item()),
        "final_position_rms": float(
            _vector_rms(result.final_residual - trajectory.final_residual).item()
        ),
        "residual_rms": float(rollout_rms.item()),
        "teacher_residual_rms": float(teacher_rms.item()),
        "amplitude_ratio": float(
            (rollout_rms / teacher_rms.clamp(min=1e-12)).item()
        ),
        "far_near_amplitude_ratio": float(
            (amplitude[:, far].mean() / amplitude[:, near].mean().clamp(min=1e-12)).item()
        ),
        "final_residual_rms": float(_vector_rms(result.final_residual).item()),
        "hold_entry_velocity_rms": float(entry_velocity.item()),
        "final_velocity_rms": float(final_velocity.item()),
        "hold_velocity_decay_ratio": float(
            (final_velocity / entry_velocity.clamp(min=1e-12)).item()
        ),
        "lbs_improvement_fraction": float(
            (1 - _vector_rms(error) / teacher_rms.clamp(min=1e-12)).item()
        ),
        "density_acceleration_rms": result.density_acceleration_rms,
        "density_acceleration_max": result.density_acceleration_max,
        "density_cap_fraction": result.density_cap_fraction,
        "finite": bool(result.finite),
    }
    report.update(measure_h8_compression_error(
        result.residual,
        trajectory,
        graph,
        float(volume.metadata["pitch"]),
        config,
    ))
    return report


def acceptance_h8(teacher, one_step, hybrid, backbone):
    """Apply H8's frozen streaming safety and conditional causal gates."""
    position_reduction = 1 - hybrid["position_rms"] / max(
        backbone["position_rms"], 1e-12
    )
    compression_reduction = 1 - hybrid["compression_error_rms"] / max(
        backbone["compression_error_rms"], 1e-12
    )
    causal_eligible = bool(
        teacher["density_acceleration_rms"] >= 0.02
        and backbone["position_rms"] >= 0.0002
    )
    softness_profile_ratio = (
        hybrid["far_near_amplitude_ratio"]
        / max(teacher["far_near_amplitude_ratio"], 1e-12)
    )
    velocity_limit = max(0.35 * hybrid["hold_entry_velocity_rms"], 0.005)
    safety = {
        "teacher_finite": teacher["finite"],
        "teacher_residual_bounded": 0.0005 <= teacher["residual_rms"] <= 0.075,
        "teacher_density_bounded": teacher["density_acceleration_max"] <= 12.0001,
        "one_step_learning": one_step["acceleration_nrmse"] <= 0.15,
        "rollout_rms": hybrid["position_rms"] <= 0.004,
        "rollout_p99": hybrid["position_p99"] <= 0.012,
        "rollout_max": hybrid["position_max"] <= 0.040,
        "rollout_amplitude": 0.70 <= hybrid["amplitude_ratio"] <= 1.30,
        "rollout_softness": 0.90 <= softness_profile_ratio <= 1.10,
        "rollout_final_residual": hybrid["final_residual_rms"] <= 0.020,
        "rollout_settles": hybrid["final_velocity_rms"] <= velocity_limit,
        "rollout_finite": hybrid["finite"],
        "predicted_density_bounded": hybrid["density_acceleration_max"] <= 12.0001,
    }
    causal = {
        "position_causal": (not causal_eligible) or position_reduction >= 0.60,
        "compression_causal": (
            (not causal_eligible) or compression_reduction >= 0.50
        ),
    }
    return {
        **safety,
        **causal,
        "causal_eligible": causal_eligible,
        "position_error_reduction": position_reduction,
        "compression_error_reduction": compression_reduction,
        "softness_profile_ratio": softness_profile_ratio,
        "settling_velocity_limit": velocity_limit,
        "safety_pass": all(safety.values()),
        "causal_pass": all(causal.values()),
        "pass": all(safety.values()) and all(causal.values()),
    }


def aggregate_h8(variant_reports, minimum_causal_variants=0):
    """Require universal safety/eligible causality and a final eligibility floor."""
    variants = list(variant_reports)
    eligible = [item for item in variants if item["causal_eligible"]]
    safety = all(item["safety_pass"] for item in variants)
    causal = all(item["causal_pass"] for item in eligible)
    enough = len(eligible) >= int(minimum_causal_variants)
    return {
        "variant_count": len(variants),
        "causal_eligible_variants": len(eligible),
        "minimum_causal_variants": int(minimum_causal_variants),
        "all_safety": safety,
        "all_eligible_causal": causal,
        "enough_causal_variants": enough,
        "pass": safety and causal and enough,
    }
