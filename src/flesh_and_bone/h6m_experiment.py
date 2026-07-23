"""Evaluate frozen H5U rules on predeclared unseen motion cycles."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import torch

from .experiment import _device
from .flesh_rollout import rollout_flesh_rule
from .flesh_rule import FleshResidualRule
from .flesh_teacher import (
    ElasticTeacherConfig,
    build_voxel_graph,
    simulate_teacher_from_lbs,
    volume_lbs_cycle,
)
from .h4_render import load_base_color, render_colored_splats, sample_texture
from .h4_volume import load_h4_volume
from .h5_metrics import measure_rollout, measure_teacher
from .h6m_metrics import acceptance_h6m, measure_frozen_feature_shift
from .motion_variants import controlled_motion_cycles
from .render import save_contact_sheet
from .rig_asset import load_rig_asset


@dataclass(frozen=True)
class H6MConfig:
    archive_path: str = "model/Meshy_AI_Blonde_female_mechani_biped.zip"
    rig_asset_path: str = "model/derived/meshy_blonde_h4_rig.npz"
    volume_path: str = "model/derived/meshy_blonde_h4_volume_p0125.npz"
    checkpoint_directory: str = "experiments/runs/h5u_final"
    seeds: tuple[int, ...] = (7, 19, 31)
    hidden_channels: int = 96
    cycles: int = 10
    feature_sample_examples: int = 131072
    device: str = "cpu"
    render_seed: int = 7
    render_motions: tuple[str, ...] = (
        "reverse", "half_speed", "fast_1p526", "walk_then_hold"
    )
    image_size: int = 720
    render_splat_radius_scale: float = 0.50
    render_opacity: float = 0.72


def _checkpoint_path(config, seed):
    return (
        Path(config.checkpoint_directory)
        / f"seed{seed}"
        / "flesh_rule.pt"
    )


def _load_rule(config, seed, device):
    path = _checkpoint_path(config, seed)
    if not path.exists():
        raise FileNotFoundError(f"missing frozen H5U checkpoint: {path}")
    rule = FleshResidualRule(config.hidden_channels).to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    rule.load_state_dict(checkpoint)
    rule.eval()
    return rule


def _render_motion(output, motion, volume, trajectory, learned, colors,
                   config):
    phase_count = motion.lbs_positions.shape[0]
    milestone_count = min(5, phase_count)
    milestones = torch.linspace(
        0, phase_count - 1, milestone_count
    ).round().to(torch.long).unique().tolist()
    teacher_frames = []
    learned_frames = []
    teacher = trajectory.residual[:, 0].detach().cpu()
    lbs = motion.lbs_positions.detach().cpu()
    bones = motion.bone_endpoints.detach().cpu()
    learned = learned.detach().cpu()
    splat_radius = (
        config.render_splat_radius_scale * volume.metadata["pitch"]
    )
    splat_scale = volume.splat_scale.detach().cpu().numpy()
    for phase in milestones:
        common = {
            "colors": colors,
            "bone_endpoints": bones[phase].numpy(),
            "splat_radius": splat_radius,
            "splat_scale": splat_scale,
            "size": config.image_size,
            "opacity": config.render_opacity,
        }
        teacher_frames.append(render_colored_splats(
            (lbs[phase] + teacher[phase]).numpy(),
            label=f"{motion.name} teacher phase {phase:02d}",
            **common,
        ))
        learned_frames.append(render_colored_splats(
            (lbs[phase] + learned[phase]).numpy(),
            label=f"{motion.name} frozen seed {config.render_seed} phase {phase:02d}",
            **common,
        ))
    save_contact_sheet(
        teacher_frames,
        output / motion.name / "teacher_contact_sheet.png",
        columns=len(teacher_frames),
    )
    save_contact_sheet(
        learned_frames,
        output / motion.name / "learned_contact_sheet.png",
        columns=len(learned_frames),
    )


def _write_report(output, report):
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


def run_h6m(output_directory, config=None, teacher_config=None):
    """Run replay plus four frozen-checkpoint generalization evaluations."""
    config = config or H6MConfig()
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
    motions = controlled_motion_cycles(
        base_lbs, asset.bone_endpoints[:-1]
    )
    colors = None
    if config.render_motions:
        texture = load_base_color(config.archive_path)
        colors = sample_texture(texture, volume.uv.detach().cpu().numpy())

    report = {
        "experiment": "H6M",
        "description": "frozen H5U rule under unseen temporal forcing",
        "config": asdict(config),
        "teacher_config": asdict(teacher_config),
        "motions": [],
        "acceptance": {
            "calibration_all_seeds": False,
            "controlled_all_motions_all_seeds": False,
            "pass": False,
        },
        "elapsed_seconds": 0.0,
    }
    _write_report(output, report)

    for motion in motions:
        motion_started = time.time()
        motion_output = output / motion.name
        motion_output.mkdir(parents=True, exist_ok=True)
        print(
            f"H6M {motion.name}: {motion.lbs_positions.shape[0]} phases",
            flush=True,
        )
        trajectory = simulate_teacher_from_lbs(
            motion.lbs_positions,
            volume,
            graph,
            config=teacher_config,
        )
        teacher_metrics = measure_teacher(trajectory, volume, graph)
        seed_reports = []
        render_learned = None
        for seed in config.seeds:
            seed_started = time.time()
            rule = _load_rule(config, seed, device)
            feature_shift = measure_frozen_feature_shift(
                trajectory,
                volume,
                rule,
                sample_examples=config.feature_sample_examples,
            )
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
            if (
                seed == config.render_seed
                and motion.name in config.render_motions
            ):
                render_learned = learned_rollout[-1].detach().cpu()
            del learned_rollout
            neighbor_rollout, _, _ = rollout_flesh_rule(
                rule,
                trajectory,
                volume,
                graph,
                teacher_config,
                cycles=config.cycles,
                neighbor_enabled=False,
            )
            neighbor_blind = measure_rollout(
                neighbor_rollout, trajectory, volume, graph
            )
            del neighbor_rollout
            gates = acceptance_h6m(
                teacher_metrics, learned, neighbor_blind
            )
            seed_report = {
                "seed": int(seed),
                "feature_shift": feature_shift,
                "learned_rollout": learned,
                "neighbor_blind_rollout": neighbor_blind,
                "acceptance": gates,
                "elapsed_seconds": time.time() - seed_started,
            }
            (motion_output / f"seed{seed}.json").write_text(
                json.dumps(seed_report, indent=2) + "\n"
            )
            seed_reports.append(seed_report)
            print(
                f"  seed {seed}: {'PASS' if gates['pass'] else 'FAIL'} "
                f"rms={learned['position_rms']:.6f} "
                f"p99={learned['position_p99']:.6f} "
                f"max={learned['position_max']:.6f}",
                flush=True,
            )
            del rule
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if render_learned is not None:
            _render_motion(
                output,
                motion,
                volume,
                trajectory,
                render_learned,
                colors,
                config,
            )
        motion_report = {
            "name": motion.name,
            "description": motion.description,
            "phase_count": int(motion.lbs_positions.shape[0]),
            "teacher": teacher_metrics,
            "seeds": seed_reports,
            "acceptance": {
                "all_seeds": all(
                    seed["acceptance"]["pass"] for seed in seed_reports
                )
            },
            "elapsed_seconds": time.time() - motion_started,
        }
        (motion_output / "metrics.json").write_text(
            json.dumps(motion_report, indent=2) + "\n"
        )
        report["motions"].append(motion_report)
        report["acceptance"]["calibration_all_seeds"] = bool(
            report["motions"][0]["acceptance"]["all_seeds"]
        )
        controlled = report["motions"][1:]
        report["acceptance"]["controlled_all_motions_all_seeds"] = bool(
            len(controlled) == 4
            and all(item["acceptance"]["all_seeds"] for item in controlled)
        )
        report["acceptance"]["pass"] = bool(
            report["acceptance"]["calibration_all_seeds"]
            and report["acceptance"]["controlled_all_motions_all_seeds"]
        )
        report["elapsed_seconds"] = time.time() - started
        _write_report(output, report)
        del trajectory
        if device.type == "cuda":
            torch.cuda.empty_cache()

    verdict = "PASS" if report["acceptance"]["pass"] else "FAIL"
    summary = [f"# H6M run: {verdict}", ""]
    for motion in report["motions"]:
        values = [
            round(seed["learned_rollout"]["position_rms"] * 1000, 3)
            for seed in motion["seeds"]
        ]
        summary.append(
            f"- `{motion['name']}`: all seeds "
            f"`{motion['acceptance']['all_seeds']}`, RMS mm `{values}`"
        )
    summary.extend([
        "",
        f"- Acceptance: `{json.dumps(report['acceptance'], sort_keys=True)}`",
        "",
    ])
    (output / "RUN.md").write_text("\n".join(summary))
    return report
