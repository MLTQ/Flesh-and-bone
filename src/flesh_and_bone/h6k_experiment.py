"""Evaluate frozen MLP and constitutive flesh rules on a Kimodo motion."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch

from .constitutive_rule import fit_constitutive_rule
from .experiment import _device
from .flesh_rollout import rollout_flesh_rule
from .flesh_rule import FleshResidualRule
from .flesh_teacher import (
    ElasticTeacherConfig,
    build_voxel_graph,
    simulate_teacher,
    simulate_teacher_from_lbs,
)
from .h4_render import load_base_color, render_colored_splats, sample_texture
from .h4_volume import load_h4_volume
from .h5_metrics import measure_rollout, measure_teacher
from .h6m_metrics import acceptance_h6m, measure_frozen_feature_shift
from .render import save_contact_sheet
from .rig_asset import load_rig_asset


@dataclass(frozen=True)
class H6KConfig:
    archive_path: str = "model/Meshy_AI_Blonde_female_mechani_biped.zip"
    rig_asset_path: str = "model/derived/meshy_blonde_h4_rig.npz"
    volume_path: str = "model/derived/meshy_blonde_h4_volume_p0125.npz"
    motion_path: str = (
        "experiments/runs/h6k_assets/kimodo_weight_shift_cycle.npz"
    )
    checkpoint_directory: str = "experiments/runs/h5u_final"
    seeds: tuple[int, ...] = (7, 19, 31)
    hidden_channels: int = 96
    cycles: int = 10
    feature_sample_examples: int = 131072
    device: str = "cpu"
    render_seed: int = 7
    image_size: int = 720
    render_splat_radius_scale: float = 0.50
    render_opacity: float = 0.72


def load_kimodo_cycle(path, device):
    """Load and validate the portable H6K motion artifact."""
    with np.load(Path(path), allow_pickle=False) as bundle:
        metadata = json.loads(str(bundle["metadata_json"].item()))
        if metadata.get("format") != "flesh-and-bone-h6-kimodo-motion-v1":
            raise ValueError("unsupported H6K Kimodo motion format")
        lbs = torch.as_tensor(
            np.array(bundle["lbs_positions"]),
            device=device,
            dtype=torch.float32,
        )
        bones = torch.as_tensor(
            np.array(bundle["bone_endpoints"]),
            device=device,
            dtype=torch.float32,
        )
    if lbs.ndim != 3 or lbs.shape[-1] != 3:
        raise ValueError("Kimodo LBS positions must have shape [phase, cell, 3]")
    if bones.shape[0] != lbs.shape[0] or bones.shape[-2:] != (2, 3):
        raise ValueError("Kimodo bone endpoints do not share LBS phases")
    return metadata, lbs, bones


def bridge_acceptance(metadata):
    """Apply the conversion-only H6K gates recorded before flesh rollout."""
    gates = {
        "humanoid_confidence": (
            metadata["destination_profile"]["body_type"] == "humanoid"
            and metadata["destination_profile"]["confidence"] >= 0.90
        ),
        "mapped_roles": metadata["mapped_role_count"] >= 20,
        "root_scale": 0.5 <= metadata["root_scale"] <= 1.5,
        "rest_identity_skin": (
            metadata["rest_identity_skin_max_error"] <= 1e-5
        ),
        "finite": metadata["finite"],
        "root_travel": metadata["root_xz_travel"] <= 0.15,
    }
    return {**gates, "pass": all(gates.values())}


def _load_mlp(config, seed, device):
    path = (
        Path(config.checkpoint_directory)
        / f"seed{seed}"
        / "flesh_rule.pt"
    )
    rule = FleshResidualRule(config.hidden_channels).to(device)
    rule.load_state_dict(
        torch.load(path, map_location=device, weights_only=True)
    )
    rule.eval()
    return rule


def _render_rules(output, lbs, bones, volume, teacher, mlp, constitutive,
                  colors, config):
    phases = lbs.shape[0]
    milestones = torch.linspace(0, phases - 1, 5).round().long().tolist()
    lbs = lbs.detach().cpu()
    bones = bones.detach().cpu()
    teacher = teacher.detach().cpu()
    mlp = mlp.detach().cpu()
    constitutive = constitutive.detach().cpu()
    common = {
        "colors": colors,
        "splat_radius": config.render_splat_radius_scale
        * volume.metadata["pitch"],
        "splat_scale": volume.splat_scale.detach().cpu().numpy(),
        "size": config.image_size,
        "opacity": config.render_opacity,
    }
    groups = {
        "teacher": teacher,
        f"mlp_seed{config.render_seed}": mlp,
        "constitutive": constitutive,
    }
    for name, residual in groups.items():
        frames = []
        for phase in milestones:
            frames.append(render_colored_splats(
                (lbs[phase] + residual[phase]).numpy(),
                bone_endpoints=bones[phase].numpy(),
                label=f"Kimodo {name} phase {phase:02d}",
                **common,
            ))
        save_contact_sheet(
            frames, output / f"{name}_contact_sheet.png", columns=5
        )


def run_h6k(output_directory, config=None, teacher_config=None):
    """Run external-motion bridge, MLP, and constitutive verdicts."""
    config = config or H6KConfig()
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
    metadata, lbs, bones = load_kimodo_cycle(config.motion_path, device)
    bridge = bridge_acceptance(metadata)
    trajectory = simulate_teacher_from_lbs(
        lbs, volume, graph, config=teacher_config
    )
    teacher = measure_teacher(trajectory, volume, graph)

    mlp_reports = []
    render_mlp = None
    for seed in config.seeds:
        rule = _load_mlp(config, seed, device)
        shift = measure_frozen_feature_shift(
            trajectory,
            volume,
            rule,
            sample_examples=config.feature_sample_examples,
        )
        predicted, _, _ = rollout_flesh_rule(
            rule,
            trajectory,
            volume,
            graph,
            teacher_config,
            cycles=config.cycles,
            neighbor_enabled=True,
        )
        learned = measure_rollout(predicted, trajectory, volume, graph)
        if seed == config.render_seed:
            render_mlp = predicted[-1].detach().cpu()
        del predicted
        blind_state, _, _ = rollout_flesh_rule(
            rule,
            trajectory,
            volume,
            graph,
            teacher_config,
            cycles=config.cycles,
            neighbor_enabled=False,
        )
        blind = measure_rollout(blind_state, trajectory, volume, graph)
        del blind_state
        gates = acceptance_h6m(teacher, learned, blind)
        item = {
            "seed": int(seed),
            "feature_shift": shift,
            "learned_rollout": learned,
            "neighbor_blind_rollout": blind,
            "acceptance": gates,
        }
        mlp_reports.append(item)
        (output / f"mlp_seed{seed}.json").write_text(
            json.dumps(item, indent=2) + "\n"
        )
        print(
            f"H6K MLP seed {seed}: {'PASS' if gates['pass'] else 'FAIL'} "
            f"rms={learned['position_rms']:.6f} "
            f"max={learned['position_max']:.6f}",
            flush=True,
        )
        del rule
        if device.type == "cuda":
            torch.cuda.empty_cache()

    replay_teacher = simulate_teacher(
        asset, volume, graph, config=teacher_config
    )
    constitutive_rule, constitutive_fit = fit_constitutive_rule(
        replay_teacher, volume
    )
    constitutive_state, _, _ = rollout_flesh_rule(
        constitutive_rule,
        trajectory,
        volume,
        graph,
        teacher_config,
        cycles=config.cycles,
        neighbor_enabled=True,
    )
    constitutive_metrics = measure_rollout(
        constitutive_state, trajectory, volume, graph
    )
    render_constitutive = constitutive_state[-1].detach().cpu()
    del constitutive_state
    constitutive_blind_state, _, _ = rollout_flesh_rule(
        constitutive_rule,
        trajectory,
        volume,
        graph,
        teacher_config,
        cycles=config.cycles,
        neighbor_enabled=False,
    )
    constitutive_blind = measure_rollout(
        constitutive_blind_state, trajectory, volume, graph
    )
    del constitutive_blind_state
    constitutive_gates = acceptance_h6m(
        teacher, constitutive_metrics, constitutive_blind
    )
    print(
        f"H6K constitutive: "
        f"{'PASS' if constitutive_gates['pass'] else 'FAIL'} "
        f"rms={constitutive_metrics['position_rms']:.6f} "
        f"max={constitutive_metrics['position_max']:.6f}",
        flush=True,
    )

    texture = load_base_color(config.archive_path)
    colors = sample_texture(texture, volume.uv.detach().cpu().numpy())
    _render_rules(
        output,
        lbs,
        bones,
        volume,
        trajectory.residual[:, 0],
        render_mlp,
        render_constitutive,
        colors,
        config,
    )
    mlp_pass = all(item["acceptance"]["pass"] for item in mlp_reports)
    report = {
        "experiment": "H6K",
        "description": "Kimodo ecological motion transfer",
        "config": asdict(config),
        "teacher_config": asdict(teacher_config),
        "motion_metadata": metadata,
        "bridge_acceptance": bridge,
        "teacher": teacher,
        "mlp": mlp_reports,
        "constitutive": {
            "fit": asdict(constitutive_fit),
            "learned_rollout": constitutive_metrics,
            "neighbor_blind_rollout": constitutive_blind,
            "acceptance": constitutive_gates,
        },
        "verdicts": {
            "bridge_validity": bridge["pass"],
            "mlp_ecological_generalization": mlp_pass,
            "constitutive_ecological_generalization": (
                constitutive_gates["pass"]
            ),
        },
        "elapsed_seconds": time.time() - started,
    }
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    (output / "RUN.md").write_text(
        "# H6K run\n\n"
        + f"- Bridge validity: `{bridge['pass']}`\n"
        + f"- MLP ecological generalization: `{mlp_pass}`\n"
        + "- Constitutive ecological generalization: "
        + f"`{constitutive_gates['pass']}`\n"
    )
    return report
