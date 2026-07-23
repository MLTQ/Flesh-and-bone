"""Frozen H7C mechanics over sealed, nonperiodic Kimodo motion streams."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch

from .density_rule import density_acceleration_nrmse
from .experiment import _device
from .flesh_teacher import build_voxel_graph
from .h4_volume import load_h4_volume
from .h7_experiment import (
    H7Config,
    _load_backbone,
    _load_checkpoint,
    h7c_teacher_config,
    h7c_training_config,
)
from .h8_metrics import acceptance_h8, aggregate_h8, measure_h8_rollout, measure_h8_teacher
from .h8_streaming import (
    build_motion_stream,
    rollout_streaming_density,
    simulate_streaming_teacher,
)
from .retarget_skin import canonical_motion_skin
from .rig_asset import linear_skin, load_rig_asset


QUALIFICATION_SPECS = {
    "standing_wave": {
        "prompt": "A gentle wave while standing in place, feet planted, upright posture",
        "seed": 1847,
    },
    "two_step": {
        "prompt": "A relaxed two-step dance, upright torso, balanced feet, subtle arm movement",
        "seed": 948593612,
    },
}

FINAL_SPECS = {
    "clean_walk": {
        "prompt": "A confident forward walk, natural arm swing, steady upright pelvis",
        "seed": 81273,
    },
    "side_sway": {
        "prompt": "A playful side-to-side sway with small steps and both arms visible",
        "seed": 26041,
    },
    "quick_turn": {
        "prompt": "A brisk half-turn and return to front, balanced footwork, arms following naturally",
        "seed": 73109,
    },
}


@dataclass(frozen=True)
class H8Config:
    """Frozen H8 assets, checkpoints, timing arms, seeds, and render bounds."""

    rig_asset_path: str = "model/derived/meshy_blonde_h4_rig.npz"
    volume_path: str = "model/derived/meshy_blonde_h4_volume_p0125.npz"
    h6c_metrics_path: str = "experiments/runs/h6c_final/metrics.json"
    checkpoint_directory: str = "experiments/runs/h7c_initial"
    speeds: tuple[float, ...] = (1.0, 2.0)
    hold_frames: int = 30
    seeds: tuple[int, ...] = (7, 19, 31)
    diagnostic_cells: int = 512
    render_cell_stride: int = 4
    render_frames: int = 60
    device: str = "cpu"


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _load_motion_metadata(bundle):
    if "metadata_json" not in bundle:
        raise ValueError("retargeted motion lacks metadata_json")
    return json.loads(str(bundle["metadata_json"].item()))


def _validate_motion(name, metadata, expected):
    if metadata.get("prompt") != expected["prompt"]:
        raise ValueError(f"{name} prompt does not match the frozen H8 specification")
    if int(metadata.get("seed", -1)) != int(expected["seed"]):
        raise ValueError(f"{name} seed does not match the frozen H8 specification")
    if not metadata.get("contact_ik_applied", False):
        raise ValueError(f"{name} must retain contact IK for H8")
    if int(metadata.get("fps", 0)) != 30:
        raise ValueError(f"{name} must be a 30 fps retarget artifact")


def load_h8_lbs_motion(path, asset, volume, device):
    """Load portable local rotations and skin the full dense H5U body in chunks."""
    with np.load(Path(path), allow_pickle=False) as bundle:
        metadata = _load_motion_metadata(bundle)
        local = torch.as_tensor(
            np.array(bundle["local_rotations"]), device=device, dtype=torch.float32
        )
        root = torch.as_tensor(
            np.array(bundle["root_positions"]), device=device, dtype=torch.float32
        )
    skin, bones = canonical_motion_skin(asset, local, root)
    chunks = []
    for start in range(0, skin.shape[0], 4):
        chunks.append(linear_skin(
            volume.points,
            volume.weights,
            skin[start:start + 4],
        ))
    return metadata, torch.cat(chunks), bones


def _render_state(stream, teacher, backbone, hybrid, volume, config):
    selection = torch.arange(
        0,
        volume.cell_count,
        int(config.render_cell_stride),
        device=stream.lbs_positions.device,
    )
    frame_indices = torch.linspace(
        0,
        stream.frame_count - 1,
        min(int(config.render_frames), stream.frame_count),
        device=stream.lbs_positions.device,
    ).round().to(torch.long).unique()
    return {
        "format": "flesh-and-bone-h8-render-v1",
        "frame_indices": frame_indices.cpu(),
        "selection": selection.cpu(),
        "lbs": stream.lbs_positions[frame_indices][:, selection].cpu(),
        "bones": stream.bone_endpoints[frame_indices].cpu(),
        "teacher_residual": teacher.residual[frame_indices][:, selection].cpu(),
        "backbone_residual": backbone.residual[frame_indices][:, selection].cpu(),
        "hybrid_residual": hybrid.residual[frame_indices][:, selection].cpu(),
        "source_frame_count": stream.frame_count,
        "motion_frames": stream.motion_frames,
        "hold_frames": stream.hold_frames,
        "fps": stream.fps,
        "speed_multiplier": stream.speed_multiplier,
    }


def _variant_name(motion_name, speed):
    return f"{motion_name}_{'natural' if abs(speed - 1.0) < 1e-12 else '2x'}"


def _evaluate_variant(
    motion_name,
    metadata,
    lbs,
    bones,
    speed,
    seeds,
    output,
    config,
    volume,
    graph,
    backbone_rule,
    base_config,
    teacher_config,
    training_config,
):
    name = _variant_name(motion_name, speed)
    print(f"H8 {name}: cold-start teacher", flush=True)
    stream = build_motion_stream(
        lbs,
        bones,
        speed_multiplier=speed,
        hold_frames=config.hold_frames,
        fps=30.0,
    )
    teacher = simulate_streaming_teacher(
        stream,
        volume,
        graph,
        teacher_config,
        diagnostic_cells=config.diagnostic_cells,
        diagnostic_seed=8801 + round(speed * 100),
    )
    teacher_metrics = measure_h8_teacher(teacher, volume, graph)
    first_hybrid = _load_checkpoint(
        Path(config.checkpoint_directory),
        seeds[0],
        base_config,
        training_config,
        backbone_rule,
        stream.lbs_positions.device,
    )
    print(f"H8 {name}: density-blind backbone", flush=True)
    backbone = rollout_streaming_density(
        first_hybrid,
        teacher,
        volume,
        graph,
        teacher_config,
        density_enabled=False,
    )
    backbone_metrics = measure_h8_rollout(
        backbone, teacher, volume, graph, teacher_config
    )
    seed_reports = []
    for seed in seeds:
        hybrid_rule = first_hybrid if seed == seeds[0] else _load_checkpoint(
            Path(config.checkpoint_directory),
            seed,
            base_config,
            training_config,
            backbone_rule,
            stream.lbs_positions.device,
        )
        one_step = density_acceleration_nrmse(
            hybrid_rule.density_residual, teacher.diagnostic_dataset
        )
        print(f"H8 {name}: hybrid seed {seed}", flush=True)
        hybrid = rollout_streaming_density(
            hybrid_rule,
            teacher,
            volume,
            graph,
            teacher_config,
            density_enabled=True,
        )
        hybrid_metrics = measure_h8_rollout(
            hybrid, teacher, volume, graph, teacher_config
        )
        gates = acceptance_h8(
            teacher_metrics, one_step, hybrid_metrics, backbone_metrics
        )
        seed_reports.append({
            "seed": int(seed),
            "one_step": one_step,
            "hybrid_rollout": hybrid_metrics,
            "acceptance": gates,
        })
        print(
            f"  {'PASS' if gates['pass'] else 'FAIL'} "
            f"hybrid={hybrid_metrics['position_rms'] * 1000:.3f}mm "
            f"backbone={backbone_metrics['position_rms'] * 1000:.3f}mm "
            f"settle={hybrid_metrics['final_velocity_rms'] * 1000:.2f}mm/s",
            flush=True,
        )
        if seed == seeds[0]:
            torch.save(
                _render_state(stream, teacher, backbone, hybrid, volume, config),
                output / f"{name}_render_state.pt",
            )
        del hybrid
        if seed != seeds[0]:
            del hybrid_rule
        if stream.lbs_positions.device.type == "cuda":
            torch.cuda.empty_cache()
    variant = {
        "name": name,
        "motion": motion_name,
        "speed_multiplier": float(speed),
        "source_metadata": metadata,
        "frame_count": stream.frame_count,
        "motion_frames": stream.motion_frames,
        "hold_frames": stream.hold_frames,
        "teacher": teacher_metrics,
        "backbone_rollout": backbone_metrics,
        "seeds": seed_reports,
        "all_seeds_safety": all(
            item["acceptance"]["safety_pass"] for item in seed_reports
        ),
        "all_seeds_pass": all(item["acceptance"]["pass"] for item in seed_reports),
        "causal_eligible": bool(seed_reports[0]["acceptance"]["causal_eligible"]),
        "all_eligible_causal": all(
            item["acceptance"]["causal_pass"] for item in seed_reports
        ),
    }
    _write_json(output / f"{name}.json", variant)
    del teacher, backbone, first_hybrid, stream
    if lbs.device.type == "cuda":
        torch.cuda.empty_cache()
    return variant


def run_h8_stage(stage, motion_paths, output_directory, config=None):
    """Run qualification or sealed final H8 streams with frozen checkpoints."""
    if stage not in {"qualification", "final"}:
        raise ValueError("stage must be qualification or final")
    config = config or H8Config()
    expected = QUALIFICATION_SPECS if stage == "qualification" else FINAL_SPECS
    if set(motion_paths) != set(expected):
        raise ValueError(f"{stage} motions must be exactly {sorted(expected)}")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    qualification_path = output / "qualification.json"
    if stage == "final":
        if not qualification_path.exists():
            raise RuntimeError("H8 final suite remains sealed: qualification is absent")
        qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
        if not qualification["acceptance"]["pass"]:
            raise RuntimeError("H8 final suite remains sealed: qualification failed")
    started = time.time()
    device = _device(config.device)
    asset = load_rig_asset(config.rig_asset_path, device=device, dtype=torch.float32)
    volume = load_h4_volume(config.volume_path, device=device, dtype=torch.float32)
    graph = build_voxel_graph(volume.points, volume.metadata["pitch"])
    base_config = H7Config(
        h6c_metrics_path=config.h6c_metrics_path,
        seeds=config.seeds,
        cycles=1,
        device=config.device,
    )
    backbone_rule, backbone_provenance = _load_backbone(base_config, device)
    teacher_config = h7c_teacher_config()
    training_config = h7c_training_config()
    seeds = (config.seeds[0],) if stage == "qualification" else config.seeds
    variants = []
    sources = {}
    for motion_name in expected:
        metadata, lbs, bones = load_h8_lbs_motion(
            motion_paths[motion_name], asset, volume, device
        )
        _validate_motion(motion_name, metadata, expected[motion_name])
        sources[motion_name] = {
            "path": str(Path(motion_paths[motion_name])),
            "metadata": metadata,
        }
        for speed in config.speeds:
            variants.append(_evaluate_variant(
                motion_name,
                metadata,
                lbs,
                bones,
                speed,
                seeds,
                output,
                config,
                volume,
                graph,
                backbone_rule,
                base_config,
                teacher_config,
                training_config,
            ))
        del lbs, bones
        if device.type == "cuda":
            torch.cuda.empty_cache()
    variant_acceptance = [
        {
            "safety_pass": variant["all_seeds_safety"],
            "causal_pass": variant["all_eligible_causal"],
            "causal_eligible": variant["causal_eligible"],
        }
        for variant in variants
    ]
    minimum = 0 if stage == "qualification" else 3
    acceptance = aggregate_h8(variant_acceptance, minimum_causal_variants=minimum)
    report = {
        "experiment": "H8",
        "stage": stage,
        "description": "cold-start nonperiodic Kimodo flesh transfer",
        "config": asdict(config),
        "teacher_config": asdict(teacher_config),
        "training_config": asdict(training_config),
        "backbone": backbone_provenance,
        "sources": sources,
        "variants": variants,
        "acceptance": acceptance,
        "elapsed_seconds": time.time() - started,
    }
    destination = qualification_path if stage == "qualification" else output / "metrics.json"
    _write_json(destination, report)
    lines = [f"# H8 {stage}: {'PASS' if acceptance['pass'] else 'FAIL'}", ""]
    for variant in variants:
        values = [
            round(item["hybrid_rollout"]["position_rms"] * 1000, 3)
            for item in variant["seeds"]
        ]
        lines.append(
            f"- `{variant['name']}` pass `{variant['all_seeds_pass']}`, "
            f"causal `{variant['causal_eligible']}`, RMS mm `{values}`"
        )
    lines.append("")
    (output / f"{stage.upper()}_RUN.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return report
