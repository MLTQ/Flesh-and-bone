"""Staged H7 bounded-density qualification and final holdout experiment."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import torch

from .constitutive_rule import ConstitutiveFleshRule
from .density_rollout import rollout_hybrid_density
from .density_rule import (
    BoundedDensityResidual,
    DensityStateDataset,
    DensityTrainingConfig,
    HybridDensityRule,
    concatenate_density_datasets,
    density_acceleration_nrmse,
    sample_density_states,
    train_density_residual,
)
from .density_teacher import (
    DensityTeacherConfig,
    simulate_density_teacher_from_lbs,
)
from .experiment import _device
from .flesh_teacher import build_voxel_graph, volume_lbs_cycle
from .h4_volume import load_h4_volume
from .h6k_experiment import load_kimodo_cycle
from .h7_metrics import acceptance_h7, measure_density_teacher, measure_h7_rollout
from .motion_variants import controlled_motion_cycles
from .rig_asset import load_rig_asset


@dataclass(frozen=True)
class H7Config:
    """Paths, split, seeds, and rollout duration frozen for H7."""

    rig_asset_path: str = "model/derived/meshy_blonde_h4_rig.npz"
    volume_path: str = "model/derived/meshy_blonde_h4_volume_p0125.npz"
    h6c_metrics_path: str = "experiments/runs/h6c_final/metrics.json"
    kimodo_motion_path: str = (
        "experiments/runs/h6k_assets/kimodo_weight_shift_cycle.npz"
    )
    training_motions: tuple[str, ...] = (
        "walk_replay", "reverse", "walk_then_hold"
    )
    qualification_motion: str = "half_speed"
    final_motions: tuple[str, ...] = ("fast_1p526", "kimodo_weight_shift")
    seeds: tuple[int, ...] = (7, 19, 31)
    cycles: int = 20
    device: str = "cpu"


def h7c_teacher_config():
    """Return H7C's frozen 24x-original nonlinear teacher configuration."""
    return DensityTeacherConfig(
        pressure_near=720.0,
        pressure_far=1200.0,
        cohesion_near=144.0,
        cohesion_far=288.0,
    )


def h7c_training_config():
    """Return H7C's matching bounded learner configuration."""
    return DensityTrainingConfig(
        pressure_max=1440.0,
        cohesion_max=432.0,
    )


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def _load_backbone(config, device):
    report = json.loads(Path(config.h6c_metrics_path).read_text())
    coefficients = report["fit"]["coefficients"]
    return ConstitutiveFleshRule(torch.tensor(
        coefficients, device=device, dtype=torch.float32
    )), {
        "source": config.h6c_metrics_path,
        "coefficients": coefficients,
        "source_holdout_acceleration_nrmse": report["fit"][
            "holdout_acceleration_nrmse"
        ],
    }


def _common(config):
    device = _device(config.device)
    asset = load_rig_asset(
        config.rig_asset_path, device=device, dtype=torch.float32
    )
    volume = load_h4_volume(
        config.volume_path, device=device, dtype=torch.float32
    )
    graph = build_voxel_graph(volume.points, volume.metadata["pitch"])
    base_lbs, _ = volume_lbs_cycle(asset, volume)
    motions = {
        item.name: item for item in controlled_motion_cycles(
            base_lbs, asset.bone_endpoints[:-1]
        )
    }
    return device, volume, graph, motions


def _dataset_payload(dataset):
    return {
        "scalars": dataset.scalars.detach().cpu(),
        "compression_vector": dataset.compression_vector.detach().cpu(),
        "stretch_vector": dataset.stretch_vector.detach().cpu(),
        "target": dataset.target.detach().cpu(),
    }


def _load_dataset(path, device):
    payload = torch.load(path, map_location=device, weights_only=True)
    return DensityStateDataset(**payload)


def prepare_h7_training(output_directory, config=None, teacher_config=None,
                        training_config=None):
    """Generate balanced H7 train states without touching held-out motions."""
    config = config or H7Config()
    teacher_config = teacher_config or DensityTeacherConfig()
    training_config = training_config or DensityTrainingConfig()
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    device, volume, graph, motions = _common(config)
    datasets = []
    teacher_reports = {}
    for index, name in enumerate(config.training_motions):
        print(f"H7 training teacher {name}", flush=True)
        trajectory = simulate_density_teacher_from_lbs(
            motions[name].lbs_positions,
            volume,
            graph,
            config=teacher_config,
        )
        teacher_reports[name] = measure_density_teacher(
            trajectory, volume, graph
        )
        datasets.append(sample_density_states(
            trajectory,
            volume,
            training_config.examples_per_motion,
            seed=7001 + index,
            teacher_config=teacher_config,
        ))
        del trajectory
        if device.type == "cuda":
            torch.cuda.empty_cache()
    dataset = concatenate_density_datasets(datasets)
    torch.save(_dataset_payload(dataset), output / "training_dataset.pt")
    report = {
        "experiment": "H7",
        "stage": "training_data",
        "config": asdict(config),
        "teacher_config": asdict(teacher_config),
        "training_config": asdict(training_config),
        "teacher_reports": teacher_reports,
        "training_examples": dataset.example_count,
        "elapsed_seconds": time.time() - started,
    }
    _write_json(output / "training_data.json", report)
    return report


def _new_hybrid(config, training_config, backbone, device):
    residual = BoundedDensityResidual(
        hidden_channels=training_config.hidden_channels,
        pressure_max=training_config.pressure_max,
        cohesion_max=training_config.cohesion_max,
        acceleration_cap=training_config.density_acceleration_cap,
    ).to(device)
    return HybridDensityRule(backbone, residual).eval()


def _save_checkpoint(output, seed, hybrid, training_report):
    directory = output / f"seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        hybrid.density_residual.state_dict(),
        directory / "density_residual.pt",
    )
    _write_json(directory / "training.json", training_report)


def _load_checkpoint(output, seed, config, training_config, backbone, device):
    hybrid = _new_hybrid(config, training_config, backbone, device)
    state = torch.load(
        output / f"seed{seed}" / "density_residual.pt",
        map_location=device,
        weights_only=True,
    )
    hybrid.density_residual.load_state_dict(state)
    return hybrid.eval()


def _fit_seed(output, seed, dataset, config, training_config, backbone, device):
    residual, training_report = train_density_residual(
        dataset, seed=seed, config=training_config
    )
    hybrid = HybridDensityRule(backbone, residual).eval()
    _save_checkpoint(output, seed, hybrid, training_report)
    return hybrid, training_report


def _evaluate_motion(hybrid, trajectory, volume, graph, teacher_config,
                     cycles, one_step):
    teacher = measure_density_teacher(trajectory, volume, graph)
    backbone_result = rollout_hybrid_density(
        hybrid,
        trajectory,
        volume,
        graph,
        teacher_config,
        cycles=cycles,
        density_enabled=False,
    )
    backbone = measure_h7_rollout(
        backbone_result, trajectory, volume, graph, teacher_config
    )
    del backbone_result
    hybrid_result = rollout_hybrid_density(
        hybrid,
        trajectory,
        volume,
        graph,
        teacher_config,
        cycles=cycles,
        density_enabled=True,
    )
    learned = measure_h7_rollout(
        hybrid_result, trajectory, volume, graph, teacher_config
    )
    gates = acceptance_h7(teacher, one_step, learned, backbone)
    return teacher, backbone, learned, gates, hybrid_result.residual[-1].cpu()


def run_h7_qualification(output_directory, config=None, teacher_config=None,
                         training_config=None):
    """Train seed 7 and open only the predeclared half-speed qualification."""
    config = config or H7Config()
    teacher_config = teacher_config or DensityTeacherConfig()
    training_config = training_config or DensityTrainingConfig()
    output = Path(output_directory)
    data_path = output / "training_dataset.pt"
    if not data_path.exists():
        prepare_h7_training(
            output, config, teacher_config, training_config
        )
    started = time.time()
    device, volume, graph, motions = _common(config)
    dataset = _load_dataset(data_path, device)
    backbone, backbone_provenance = _load_backbone(config, device)
    seed = config.seeds[0]
    hybrid, training_report = _fit_seed(
        output, seed, dataset, config, training_config, backbone, device
    )
    train_gate = training_report["acceleration_nrmse"] <= 0.10
    del dataset
    print(
        f"H7 seed {seed} train NRMSE="
        f"{training_report['acceleration_nrmse']:.5f}",
        flush=True,
    )
    motion = motions[config.qualification_motion]
    trajectory = simulate_density_teacher_from_lbs(
        motion.lbs_positions, volume, graph, config=teacher_config
    )
    qualification_dataset = sample_density_states(
        trajectory, volume, 300000, 7707, teacher_config
    )
    one_step = density_acceleration_nrmse(
        hybrid.density_residual, qualification_dataset
    )
    teacher, backbone_metrics, learned, gates, render_state = _evaluate_motion(
        hybrid,
        trajectory,
        volume,
        graph,
        teacher_config,
        config.cycles,
        one_step,
    )
    aggregate = bool(train_gate and gates["pass"])
    report = {
        "experiment": "H7",
        "stage": "qualification",
        "motion": config.qualification_motion,
        "seed": int(seed),
        "config": asdict(config),
        "teacher_config": asdict(teacher_config),
        "training_config": asdict(training_config),
        "backbone": backbone_provenance,
        "training": training_report,
        "training_nrmse_gate": train_gate,
        "qualification_one_step": one_step,
        "teacher": teacher,
        "backbone_rollout": backbone_metrics,
        "hybrid_rollout": learned,
        "acceptance": {**gates, "pass": aggregate},
        "elapsed_seconds": time.time() - started,
    }
    _write_json(output / "qualification.json", report)
    torch.save(render_state, output / "qualification_render_state.pt")
    print(
        f"H7 qualification {'PASS' if aggregate else 'FAIL'} "
        f"hybrid={learned['position_rms'] * 1000:.3f}mm "
        f"backbone={backbone_metrics['position_rms'] * 1000:.3f}mm",
        flush=True,
    )
    return report


def run_h7_final(output_directory, config=None, teacher_config=None,
                 training_config=None):
    """After a passing qualification, train all seeds and open final holdouts."""
    config = config or H7Config()
    teacher_config = teacher_config or DensityTeacherConfig()
    training_config = training_config or DensityTrainingConfig()
    output = Path(output_directory)
    qualification = json.loads((output / "qualification.json").read_text())
    if not qualification["acceptance"]["pass"]:
        raise RuntimeError("H7 final holdouts remain sealed: qualification failed")
    started = time.time()
    device, volume, graph, motions = _common(config)
    _, kimodo_lbs, _ = load_kimodo_cycle(config.kimodo_motion_path, device)
    final_lbs = {
        "fast_1p526": motions["fast_1p526"].lbs_positions,
        "kimodo_weight_shift": kimodo_lbs,
    }
    dataset = _load_dataset(output / "training_dataset.pt", device)
    backbone, backbone_provenance = _load_backbone(config, device)
    training_reports = {
        str(config.seeds[0]): qualification["training"]
    }
    for seed in config.seeds[1:]:
        _, report = _fit_seed(
            output,
            seed,
            dataset,
            config,
            training_config,
            backbone,
            device,
        )
        training_reports[str(seed)] = report
        print(
            f"H7 seed {seed} train NRMSE={report['acceleration_nrmse']:.5f}",
            flush=True,
        )
    del dataset

    motion_reports = []
    for motion_name in config.final_motions:
        print(f"H7 final holdout {motion_name}", flush=True)
        trajectory = simulate_density_teacher_from_lbs(
            final_lbs[motion_name], volume, graph, config=teacher_config
        )
        diagnostic_dataset = sample_density_states(
            trajectory, volume, 300000, 7900, teacher_config
        )
        teacher_metrics = measure_density_teacher(trajectory, volume, graph)
        seed_reports = []
        backbone_metrics = None
        for seed in config.seeds:
            hybrid = _load_checkpoint(
                output, seed, config, training_config, backbone, device
            )
            holdout_one_step = density_acceleration_nrmse(
                hybrid.density_residual, diagnostic_dataset
            )
            if backbone_metrics is None:
                backbone_result = rollout_hybrid_density(
                    hybrid,
                    trajectory,
                    volume,
                    graph,
                    teacher_config,
                    cycles=config.cycles,
                    density_enabled=False,
                )
                backbone_metrics = measure_h7_rollout(
                    backbone_result,
                    trajectory,
                    volume,
                    graph,
                    teacher_config,
                )
                del backbone_result
            hybrid_result = rollout_hybrid_density(
                hybrid,
                trajectory,
                volume,
                graph,
                teacher_config,
                cycles=config.cycles,
                density_enabled=True,
            )
            learned = measure_h7_rollout(
                hybrid_result, trajectory, volume, graph, teacher_config
            )
            training_report = training_reports[str(seed)]
            training_gate = training_report["acceleration_nrmse"] <= 0.10
            gates = acceptance_h7(
                teacher_metrics,
                {"acceleration_nrmse": training_report["acceleration_nrmse"]},
                learned,
                backbone_metrics,
            )
            aggregate = bool(training_gate and gates["pass"])
            item = {
                "seed": int(seed),
                "training": training_report,
                "training_nrmse_gate": training_gate,
                "holdout_one_step_diagnostic": holdout_one_step,
                "hybrid_rollout": learned,
                "acceptance": {**gates, "pass": aggregate},
            }
            seed_reports.append(item)
            torch.save(
                hybrid_result.residual[-1].cpu(),
                output / f"{motion_name}_seed{seed}_render_state.pt",
            )
            del hybrid_result, hybrid
            print(
                f"  seed {seed}: {'PASS' if aggregate else 'FAIL'} "
                f"hybrid={learned['position_rms'] * 1000:.3f}mm "
                f"backbone={backbone_metrics['position_rms'] * 1000:.3f}mm",
                flush=True,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
        motion_report = {
            "name": motion_name,
            "phase_count": int(final_lbs[motion_name].shape[0]),
            "teacher": teacher_metrics,
            "backbone_rollout": backbone_metrics,
            "seeds": seed_reports,
            "all_seeds": all(item["acceptance"]["pass"] for item in seed_reports),
        }
        _write_json(output / f"{motion_name}.json", motion_report)
        motion_reports.append(motion_report)
        del trajectory, diagnostic_dataset
        if device.type == "cuda":
            torch.cuda.empty_cache()
    aggregate = all(item["all_seeds"] for item in motion_reports)
    report = {
        "experiment": "H7",
        "stage": "final",
        "description": "bounded hybrid density mechanics",
        "config": asdict(config),
        "teacher_config": asdict(teacher_config),
        "training_config": asdict(training_config),
        "backbone": backbone_provenance,
        "qualification": qualification["acceptance"],
        "motions": motion_reports,
        "acceptance": {"all_final_motions_all_seeds": aggregate, "pass": aggregate},
        "elapsed_seconds": time.time() - started,
    }
    _write_json(output / "metrics.json", report)
    lines = [f"# H7 run: {'PASS' if aggregate else 'FAIL'}", ""]
    lines.append(
        f"- Half-speed qualification: `{qualification['acceptance']['pass']}`"
    )
    for motion in motion_reports:
        values = [
            round(item["hybrid_rollout"]["position_rms"] * 1000, 3)
            for item in motion["seeds"]
        ]
        lines.append(
            f"- `{motion['name']}` all seeds `{motion['all_seeds']}`, "
            f"RMS mm `{values}`"
        )
    lines.append("")
    (output / "RUN.md").write_text("\n".join(lines))
    return report
