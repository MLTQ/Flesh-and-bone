"""Train and validate separately taught H5 local flesh mechanics."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import torch

from .experiment import _device
from .flesh_rollout import (
    FleshTrainingConfig,
    rollout_flesh_rule,
    train_flesh_rule,
)
from .flesh_teacher import (
    ElasticTeacherConfig,
    build_voxel_graph,
    simulate_teacher,
)
from .h4_render import load_base_color, render_colored_splats, sample_texture
from .h4_volume import load_h4_volume
from .h5_metrics import acceptance_h5, measure_rollout, measure_teacher
from .render import save_contact_sheet, save_gif
from .rig_asset import load_rig_asset


@dataclass(frozen=True)
class H5Config:
    archive_path: str = "model/Meshy_AI_Blonde_female_mechani_biped.zip"
    rig_asset_path: str = "model/derived/meshy_blonde_h4_rig.npz"
    volume_path: str = "model/derived/meshy_blonde_h4_volume.npz"
    seeds: tuple[int, ...] = (7, 19, 31)
    device: str = "cpu"
    image_size: int = 480
    render_seed: int = 7


def _render_seed(output, asset, volume, trajectory, rollout, seed, archive_path,
                 image_size):
    texture = load_base_color(archive_path)
    colors = sample_texture(texture, volume.uv.cpu().numpy())
    teacher_residual = trajectory.residual[:, 0]
    learned_residual = rollout[-1]
    teacher_frames, learned_frames = [], []
    for phase in range(trajectory.lbs_positions.shape[0]):
        bones = asset.bone_endpoints[phase].cpu().numpy()
        teacher_frames.append(render_colored_splats(
            (trajectory.lbs_positions[phase] + teacher_residual[phase])
            .cpu().numpy(),
            colors,
            bones,
            splat_radius=0.30 * volume.metadata["pitch"],
            splat_scale=volume.splat_scale.cpu().numpy(),
            size=image_size,
            opacity=0.52,
            label=f"teacher seed {seed} phase {phase:02d}",
        ))
        learned_frames.append(render_colored_splats(
            (trajectory.lbs_positions[phase] + learned_residual[phase])
            .cpu().numpy(),
            colors,
            bones,
            splat_radius=0.30 * volume.metadata["pitch"],
            splat_scale=volume.splat_scale.cpu().numpy(),
            size=image_size,
            opacity=0.52,
            label=f"learned seed {seed} phase {phase:02d}",
        ))
    save_gif(teacher_frames, output / "teacher_animation.gif")
    save_gif(learned_frames, output / "learned_animation.gif")
    milestones = [0, 7, 14, 21, 28]
    save_contact_sheet(
        [teacher_frames[index] for index in milestones],
        output / "teacher_contact_sheet.png",
        columns=5,
    )
    save_contact_sheet(
        [learned_frames[index] for index in milestones],
        output / "learned_contact_sheet.png",
        columns=5,
    )
    teacher_exaggerated = []
    learned_exaggerated = []
    for phase in milestones:
        bones = asset.bone_endpoints[phase].cpu().numpy()
        teacher_exaggerated.append(render_colored_splats(
            (trajectory.lbs_positions[phase] + 4 * teacher_residual[phase])
            .cpu().numpy(), colors, bones,
            splat_radius=0.30 * volume.metadata["pitch"],
            splat_scale=volume.splat_scale.cpu().numpy(), size=image_size,
            opacity=0.52, label=f"teacher residual 4x phase {phase:02d}",
        ))
        learned_exaggerated.append(render_colored_splats(
            (trajectory.lbs_positions[phase] + 4 * learned_residual[phase])
            .cpu().numpy(), colors, bones,
            splat_radius=0.30 * volume.metadata["pitch"],
            splat_scale=volume.splat_scale.cpu().numpy(), size=image_size,
            opacity=0.52, label=f"learned residual 4x phase {phase:02d}",
        ))
    save_contact_sheet(
        teacher_exaggerated,
        output / "teacher_residual_4x.png",
        columns=5,
    )
    save_contact_sheet(
        learned_exaggerated,
        output / "learned_residual_4x.png",
        columns=5,
    )


def run_h5(output_directory, config=None, teacher_config=None,
           training_config=None):
    """Train frozen seeds and write H5 metrics, checkpoints, and visuals."""
    config = config or H5Config()
    teacher_config = teacher_config or ElasticTeacherConfig()
    training_config = training_config or FleshTrainingConfig()
    device = _device(config.device)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    started = time.time()
    asset = load_rig_asset(
        config.rig_asset_path, device=device, dtype=torch.float32
    )
    volume = load_h4_volume(
        config.volume_path, device=device, dtype=torch.float32
    )
    graph = build_voxel_graph(volume.points, volume.metadata["pitch"])
    trajectory = simulate_teacher(
        asset, volume, graph, config=teacher_config
    )
    teacher_metrics = measure_teacher(trajectory, volume, graph)

    runs = []
    render_payload = None
    for seed in config.seeds:
        seed_output = output_directory / f"seed{seed}"
        seed_output.mkdir(parents=True, exist_ok=True)
        rule, training = train_flesh_rule(
            trajectory, volume, seed, training_config
        )
        learned_rollout, _, _ = rollout_flesh_rule(
            rule,
            trajectory,
            volume,
            graph,
            config.image_size,
            cycles=training_config.rollout_cycles,
            neighbor_enabled=True,
        )
        blind_rollout, _, _ = rollout_flesh_rule(
            rule,
            trajectory,
            volume,
            graph,
            teacher_config,
            cycles=training_config.rollout_cycles,
            neighbor_enabled=False,
        )
        learned = measure_rollout(
            learned_rollout, trajectory, volume, graph
        )
        neighbor_blind = measure_rollout(
            blind_rollout, trajectory, volume, graph
        )
        gates = acceptance_h5(
            teacher_metrics, training, learned, neighbor_blind
        )
        checkpoint = {
            key: value.detach().cpu()
            for key, value in rule.state_dict().items()
        }
        torch.save(checkpoint, seed_output / "flesh_rule.pt")
        seed_report = {
            "seed": seed,
            "training": training,
            "learned_rollout": learned,
            "neighbor_blind_rollout": neighbor_blind,
            "acceptance": gates,
        }
        (seed_output / "metrics.json").write_text(
            json.dumps(seed_report, indent=2) + "\n"
        )
        runs.append(seed_report)
        if seed == config.render_seed:
            render_payload = (learned_rollout.detach().cpu(), seed)

    if render_payload is not None:
        learned_cpu, seed = render_payload
        cpu_asset = load_rig_asset(config.rig_asset_path, dtype=torch.float32)
        cpu_volume = load_h4_volume(config.volume_path, dtype=torch.float32)
        cpu_trajectory = type(trajectory)(
            **{
                field: getattr(trajectory, field).detach().cpu()
                for field in trajectory.__dataclass_fields__
            }
        )
        _render_seed(
            output_directory,
            cpu_asset,
            cpu_volume,
            cpu_trajectory,
            learned_cpu,
            seed,
            config.archive_path,
            teacher_config,
        )
    aggregate_pass = all(run["acceptance"]["pass"] for run in runs)
    report = {
        "experiment": "H5",
        "description": "separately taught local graph-elastic flesh mechanics",
        "config": asdict(config),
        "teacher_config": asdict(teacher_config),
        "training_config": asdict(training_config),
        "teacher": teacher_metrics,
        "runs": runs,
        "acceptance": {"all_seeds": aggregate_pass, "pass": aggregate_pass},
        "elapsed_seconds": time.time() - started,
    }
    (output_directory / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    verdict = "PASS" if aggregate_pass else "FAIL"
    (output_directory / "RUN.md").write_text(f"""# H5 run: {verdict}

- Cells / directed graph edges: `{volume.cell_count}` / `{teacher_metrics['graph_directed_edges']}`
- Teacher residual RMS / far-near ratio: `{teacher_metrics['residual_rms']:.5f}` m / `{teacher_metrics['far_near_amplitude_ratio']:.3f}`
- Teacher cycle seam: `{teacher_metrics['cycle_seam_rms']:.9f}` m
- Seeds: `{list(config.seeds)}`
- Learned RMS by seed: `{[round(run['learned_rollout']['position_rms'], 6) for run in runs]}`
- Neighbor-blind RMS by seed: `{[round(run['neighbor_blind_rollout']['position_rms'], 6) for run in runs]}`
- Acceptance: `{json.dumps(report['acceptance'], sort_keys=True)}`
""")
    return report
