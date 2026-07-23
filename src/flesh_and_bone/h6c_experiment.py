"""Fit and validate H6C's structure-preserving constitutive rule."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import torch

from .constitutive_rule import COEFFICIENT_NAMES, fit_constitutive_rule
from .experiment import _device
from .flesh_rollout import rollout_flesh_rule
from .flesh_teacher import (
    ElasticTeacherConfig,
    build_voxel_graph,
    simulate_teacher_from_lbs,
    volume_lbs_cycle,
)
from .h4_render import load_base_color, render_colored_splats, sample_texture
from .h4_volume import load_h4_volume
from .h5_metrics import measure_rollout, measure_teacher
from .h6m_metrics import acceptance_h6m
from .motion_variants import controlled_motion_cycles
from .render import save_contact_sheet
from .rig_asset import load_rig_asset


@dataclass(frozen=True)
class H6CConfig:
    archive_path: str = "model/Meshy_AI_Blonde_female_mechani_biped.zip"
    rig_asset_path: str = "model/derived/meshy_blonde_h4_rig.npz"
    volume_path: str = "model/derived/meshy_blonde_h4_volume_p0125.npz"
    device: str = "cpu"
    cycles: int = 10
    image_size: int = 720
    render_splat_radius_scale: float = 0.50
    render_opacity: float = 0.72


def _coefficient_acceptance(fit):
    expected = (1.0, 0.44, 1200.0, 0.0, 1.0)
    recovered = fit.coefficients
    relative = {
        COEFFICIENT_NAMES[index]: abs(recovered[index] - expected[index])
        / abs(expected[index])
        for index in (0, 1, 2, 4)
    }
    gates = {
        "holdout_nrmse": fit.holdout_acceleration_nrmse <= 1e-4,
        "nonzero_coefficients": all(value <= 0.01 for value in relative.values()),
        "neighbor_velocity_zero": abs(recovered[3]) <= 1e-3,
    }
    return {
        "expected": dict(zip(COEFFICIENT_NAMES, expected)),
        "recovered": dict(zip(COEFFICIENT_NAMES, recovered)),
        "nonzero_relative_error": relative,
        **gates,
        "pass": all(gates.values()),
    }


def _render_fast(output, motion, volume, trajectory, learned, colors, config):
    phases = motion.lbs_positions.shape[0]
    milestones = torch.linspace(0, phases - 1, 5).round().long().tolist()
    teacher = trajectory.residual[:, 0].detach().cpu()
    lbs = motion.lbs_positions.detach().cpu()
    bones = motion.bone_endpoints.detach().cpu()
    learned = learned.detach().cpu()
    common = {
        "colors": colors,
        "splat_radius": config.render_splat_radius_scale
        * volume.metadata["pitch"],
        "splat_scale": volume.splat_scale.detach().cpu().numpy(),
        "size": config.image_size,
        "opacity": config.render_opacity,
    }
    teacher_frames, learned_frames = [], []
    for phase in milestones:
        teacher_frames.append(render_colored_splats(
            (lbs[phase] + teacher[phase]).numpy(),
            bone_endpoints=bones[phase].numpy(),
            label=f"fast teacher phase {phase:02d}",
            **common,
        ))
        learned_frames.append(render_colored_splats(
            (lbs[phase] + learned[phase]).numpy(),
            bone_endpoints=bones[phase].numpy(),
            label=f"fast fitted constitutive phase {phase:02d}",
            **common,
        ))
    save_contact_sheet(
        teacher_frames, output / "fast_teacher_contact_sheet.png", columns=5
    )
    save_contact_sheet(
        learned_frames,
        output / "fast_constitutive_contact_sheet.png",
        columns=5,
    )


def run_h6c(output_directory, config=None, teacher_config=None):
    """Identify one structured rule and run the complete H6M motion suite."""
    config = config or H6CConfig()
    teacher_config = teacher_config or ElasticTeacherConfig(
        neighbor_coupling=1200.0
    )
    device = _device(config.device)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    asset = load_rig_asset(
        config.rig_asset_path, device=device, dtype=torch.float32
    )
    volume = load_h4_volume(
        config.volume_path, device=device, dtype=torch.float32
    )
    graph = build_voxel_graph(volume.points, volume.metadata["pitch"])
    base_lbs, _ = volume_lbs_cycle(asset, volume)
    motions = controlled_motion_cycles(base_lbs, asset.bone_endpoints[:-1])
    replay_trajectory = simulate_teacher_from_lbs(
        base_lbs, volume, graph, config=teacher_config
    )
    rule, fit = fit_constitutive_rule(replay_trajectory, volume)
    coefficient_acceptance = _coefficient_acceptance(fit)
    texture = load_base_color(config.archive_path)
    colors = sample_texture(texture, volume.uv.detach().cpu().numpy())
    print(
        f"H6C coefficients={fit.coefficients} "
        f"holdout_nrmse={fit.holdout_acceleration_nrmse:.3e}",
        flush=True,
    )

    motion_reports = []
    for motion in motions:
        motion_started = time.time()
        trajectory = (
            replay_trajectory
            if motion.name == "walk_replay"
            else simulate_teacher_from_lbs(
                motion.lbs_positions,
                volume,
                graph,
                config=teacher_config,
            )
        )
        teacher = measure_teacher(trajectory, volume, graph)
        learned_rollout, _, _ = rollout_flesh_rule(
            rule,
            trajectory,
            volume,
            graph,
            teacher_config,
            cycles=config.cycles,
            neighbor_enabled=True,
        )
        learned = measure_rollout(
            learned_rollout, trajectory, volume, graph
        )
        render_state = (
            learned_rollout[-1].detach().cpu()
            if motion.name == "fast_1p526" else None
        )
        del learned_rollout
        blind_rollout, _, _ = rollout_flesh_rule(
            rule,
            trajectory,
            volume,
            graph,
            teacher_config,
            cycles=config.cycles,
            neighbor_enabled=False,
        )
        blind = measure_rollout(blind_rollout, trajectory, volume, graph)
        del blind_rollout
        gates = acceptance_h6m(teacher, learned, blind)
        motion_report = {
            "name": motion.name,
            "description": motion.description,
            "phase_count": int(motion.lbs_positions.shape[0]),
            "teacher": teacher,
            "learned_rollout": learned,
            "neighbor_blind_rollout": blind,
            "acceptance": gates,
            "elapsed_seconds": time.time() - motion_started,
        }
        motion_reports.append(motion_report)
        (output / f"{motion.name}.json").write_text(
            json.dumps(motion_report, indent=2) + "\n"
        )
        print(
            f"H6C {motion.name}: {'PASS' if gates['pass'] else 'FAIL'} "
            f"rms={learned['position_rms']:.6f} "
            f"max={learned['position_max']:.6f}",
            flush=True,
        )
        if render_state is not None:
            _render_fast(
                output,
                motion,
                volume,
                trajectory,
                render_state,
                colors,
                config,
            )
        if trajectory is not replay_trajectory:
            del trajectory
        if device.type == "cuda":
            torch.cuda.empty_cache()

    motion_pass = all(item["acceptance"]["pass"] for item in motion_reports)
    aggregate = coefficient_acceptance["pass"] and motion_pass
    report = {
        "experiment": "H6C",
        "description": "structure-preserving local constitutive identification",
        "config": asdict(config),
        "teacher_config": asdict(teacher_config),
        "fit": asdict(fit),
        "coefficient_acceptance": coefficient_acceptance,
        "motions": motion_reports,
        "acceptance": {
            "coefficients": coefficient_acceptance["pass"],
            "all_motions": motion_pass,
            "pass": aggregate,
        },
        "elapsed_seconds": time.time() - started,
    }
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    verdict = "PASS" if aggregate else "FAIL"
    lines = [
        f"# H6C run: {verdict}",
        "",
        f"- Coefficients: `{fit.coefficients}`",
        f"- Held-out acceleration NRMSE: `{fit.holdout_acceleration_nrmse:.8g}`",
    ]
    for motion in motion_reports:
        lines.append(
            f"- `{motion['name']}`: `{motion['acceptance']['pass']}`, "
            f"RMS `{motion['learned_rollout']['position_rms'] * 1000:.4f} mm`"
        )
    lines.append("")
    (output / "RUN.md").write_text("\n".join(lines))
    return report
