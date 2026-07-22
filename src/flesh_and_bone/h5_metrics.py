"""H5 teacher, rollout, ablation, and acceptance measurements."""

import torch


def measure_teacher(trajectory, volume, graph):
    """Measure periodicity, amplitude locality, coherence, and finite state."""
    residual = trajectory.residual[:, 0]
    amplitude = residual.norm(dim=-1)
    far = volume.bone_distance > 0.12
    near = volume.bone_distance < 0.05
    edge = residual[:, graph.target] - residual[:, graph.source]
    seam = trajectory.final_residual - residual[0]
    return {
        "graph_component_count": graph.component_count,
        "graph_directed_edges": int(graph.source.shape[0]),
        "graph_min_degree": float(graph.degree.min().item()),
        "graph_mean_degree": float(graph.degree.mean().item()),
        "finite": bool(
            torch.isfinite(trajectory.residual).all().item()
            and torch.isfinite(trajectory.velocity).all().item()
        ),
        "residual_rms": float(torch.sqrt(
            residual.square().sum(dim=-1).mean()
        ).item()),
        "far_near_amplitude_ratio": float(
            (amplitude[:, far].mean() / amplitude[:, near].mean()).item()
        ),
        "edge_difference_rms": float(torch.sqrt(
            edge.square().sum(dim=-1).mean()
        ).item()),
        "cycle_seam_rms": float(torch.sqrt(
            seam.square().sum(dim=-1).mean()
        ).item()),
    }


def measure_rollout(rollout, trajectory, volume, graph):
    """Compare repeated autoregressive residual cycles to the teacher cycle."""
    teacher = trajectory.residual[:, 0]
    error = rollout - teacher[None]
    distance = error.norm(dim=-1)
    amplitude = rollout.norm(dim=-1)
    teacher_rms = torch.sqrt(teacher.square().sum(dim=-1).mean())
    rollout_rms = torch.sqrt(rollout.square().sum(dim=-1).mean())
    far = volume.bone_distance > 0.12
    near = volume.bone_distance < 0.05
    if rollout.shape[0] > 1:
        cycle_drift = torch.sqrt(
            (rollout[1:, 0] - rollout[:-1, 0])
            .square().sum(dim=-1).mean(dim=1)
        )
        maximum_drift = cycle_drift.max()
    else:
        maximum_drift = rollout.new_zeros(())

    squared_edge_error = rollout.new_zeros(())
    edge_values = 0
    teacher_edge = teacher[:, graph.target] - teacher[:, graph.source]
    for cycle in range(rollout.shape[0]):
        for phase in range(rollout.shape[1]):
            predicted_edge = (
                rollout[cycle, phase, graph.target]
                - rollout[cycle, phase, graph.source]
            )
            squared_edge_error += (
                predicted_edge - teacher_edge[phase]
            ).square().sum()
            edge_values += predicted_edge.numel()
    return {
        "position_rms": float(torch.sqrt(error.square().sum(dim=-1).mean()).item()),
        "position_p99": float(torch.quantile(distance, 0.99).item()),
        "position_max": float(distance.max().item()),
        "residual_rms": float(rollout_rms.item()),
        "teacher_residual_rms": float(teacher_rms.item()),
        "amplitude_ratio": float((rollout_rms / teacher_rms).item()),
        "far_near_amplitude_ratio": float(
            (amplitude[..., far].mean() / amplitude[..., near].mean()).item()
        ),
        "phase_zero_cycle_drift_rms": float(maximum_drift.item()),
        "edge_strain_error_rms": float(torch.sqrt(
            squared_edge_error / max(edge_values, 1)
        ).item()),
        "lbs_improvement_fraction": float(
            (1 - torch.sqrt(error.square().sum(dim=-1).mean()) / teacher_rms)
            .item()
        ),
        "finite": bool(torch.isfinite(rollout).all().item()),
    }


def acceptance_h5(teacher, training, learned, neighbor_blind):
    """Apply H5's predeclared teacher, learning, rollout, and control gates."""
    neighbor_causal = (
        neighbor_blind["position_rms"] >= 1.20 * learned["position_rms"]
        or neighbor_blind["edge_strain_error_rms"]
        >= 1.25 * learned["edge_strain_error_rms"]
    )
    gates = {
        "graph_connected": teacher["graph_component_count"] == 1,
        "graph_no_isolates": teacher["graph_min_degree"] >= 1,
        "teacher_finite": teacher["finite"],
        "teacher_periodic": teacher["cycle_seam_rms"] <= 5e-4,
        "teacher_nontrivial": 0.002 <= teacher["residual_rms"] <= 0.030,
        "teacher_softness": teacher["far_near_amplitude_ratio"] >= 1.35,
        "teacher_coherence": teacher["edge_difference_rms"] < 0.020,
        "one_step_learning": training["holdout_acceleration_nrmse"] <= 0.15,
        "rollout_rms": learned["position_rms"] <= 0.004,
        "rollout_p99": learned["position_p99"] <= 0.012,
        "rollout_max": learned["position_max"] <= 0.040,
        "rollout_amplitude": 0.75 <= learned["amplitude_ratio"] <= 1.25,
        "rollout_softness": learned["far_near_amplitude_ratio"] >= 1.25,
        "rollout_drift": learned["phase_zero_cycle_drift_rms"] <= 0.003,
        "rollout_finite": learned["finite"],
        "neighbor_causal": neighbor_causal,
        "beats_lbs": learned["lbs_improvement_fraction"] >= 0.35,
    }
    return {**gates, "pass": all(gates.values())}
