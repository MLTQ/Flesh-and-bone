"""Predeclared teacher, compression, and acceptance metrics for H7."""

import torch

from .h5_metrics import measure_rollout, measure_teacher


def measure_density_teacher(trajectory, volume, graph):
    """Extend the shared teacher report with nonlinear-force diagnostics."""
    report = measure_teacher(trajectory, volume, graph)
    norm = trajectory.density_acceleration.norm(dim=-1)
    report.update({
        "density_acceleration_rms": float(torch.sqrt(
            norm.square().mean()
        ).item()),
        "density_acceleration_max": float(norm.max().item()),
        "compression_rms_mean": float(
            trajectory.compression_rms.mean().item()
        ),
        "stretch_rms_mean": float(trajectory.stretch_rms.mean().item()),
    })
    return report


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


def measure_compression_error(rollout, trajectory, graph, pitch, config,
                              edge_stride=8):
    """Compare sampled local compression excess across every rollout cycle."""
    source = graph.source[::int(edge_stride)]
    target = graph.target[::int(edge_stride)]
    squared_error = rollout.new_zeros(())
    squared_target = rollout.new_zeros(())
    count = 0
    teacher_mean_sum = rollout.new_zeros(())
    for phase in range(rollout.shape[1]):
        teacher_value = _compression_excess(
            trajectory.lbs_positions[phase],
            trajectory.residual[phase, 0],
            source,
            target,
            pitch,
            config,
        )
        teacher_mean_sum += teacher_value.sum()
        for cycle in range(rollout.shape[0]):
            prediction = _compression_excess(
                trajectory.lbs_positions[phase],
                rollout[cycle, phase],
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
        "compression_target_mean": float(
            teacher_mean_sum / max(
                rollout.shape[1] * source.numel(), 1
            )
        ),
        "sampled_directed_edges": int(source.numel()),
    }


def measure_h7_rollout(result, trajectory, volume, graph, config):
    """Combine inherited position metrics, compression error, and safeguards."""
    report = measure_rollout(result.residual, trajectory, volume, graph)
    report.update(measure_compression_error(
        result.residual,
        trajectory,
        graph,
        float(volume.metadata["pitch"]),
        config,
    ))
    report.update({
        "density_acceleration_rms": result.density_acceleration_rms,
        "density_acceleration_max": result.density_acceleration_max,
        "density_cap_fraction": result.density_cap_fraction,
        "rollout_integrator_finite": result.finite,
    })
    return report


def acceptance_h7(teacher, one_step, hybrid, backbone):
    """Apply the H7 gates frozen before the first qualification run."""
    position_reduction = 1 - hybrid["position_rms"] / max(
        backbone["position_rms"], 1e-12
    )
    compression_reduction = 1 - hybrid["compression_error_rms"] / max(
        backbone["compression_error_rms"], 1e-12
    )
    gates = {
        "teacher_finite": teacher["finite"],
        "teacher_periodic": teacher["cycle_seam_rms"] <= 5e-4,
        "teacher_residual_bounded": (
            0.001 <= teacher["residual_rms"] <= 0.075
        ),
        "teacher_density_nontrivial": (
            0.02 <= teacher["density_acceleration_rms"] <= 4.0
        ),
        "teacher_density_bounded": (
            teacher["density_acceleration_max"] <= 12.0001
        ),
        "one_step_learning": one_step["acceleration_nrmse"] <= 0.15,
        "rollout_rms": hybrid["position_rms"] <= 0.004,
        "rollout_p99": hybrid["position_p99"] <= 0.012,
        "rollout_max": hybrid["position_max"] <= 0.040,
        "rollout_amplitude": 0.75 <= hybrid["amplitude_ratio"] <= 1.25,
        "rollout_softness": hybrid["far_near_amplitude_ratio"] >= 1.25,
        "rollout_drift": hybrid["phase_zero_cycle_drift_rms"] <= 0.003,
        "rollout_finite": (
            hybrid["finite"] and hybrid["rollout_integrator_finite"]
        ),
        "predicted_density_bounded": (
            hybrid["density_acceleration_max"] <= 12.0001
        ),
        "nonvacuous_backbone": backbone["position_rms"] >= 0.0002,
        "position_causal": position_reduction >= 0.60,
        "compression_causal": compression_reduction >= 0.50,
    }
    return {
        **gates,
        "position_error_reduction": position_reduction,
        "compression_error_reduction": compression_reduction,
        "pass": all(gates.values()),
    }
