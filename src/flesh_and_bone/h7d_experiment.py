"""Frozen-checkpoint ecological stability and Kimodo excitation audit."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import torch

from .density_rollout import rollout_hybrid_density
from .density_rule import density_acceleration_nrmse, sample_density_states
from .density_teacher import simulate_density_teacher_from_lbs
from .h6k_experiment import load_kimodo_cycle
from .h7_experiment import (
    H7Config,
    _common,
    _load_backbone,
    _load_checkpoint,
    h7c_teacher_config,
    h7c_training_config,
)
from .h7_metrics import acceptance_h7, measure_density_teacher, measure_h7_rollout
from .motion_variants import periodic_catmull_rom


@dataclass(frozen=True)
class H7DConfig:
    """Frozen source checkpoint, stress ratio, and evaluation duration."""

    checkpoint_directory: str = "experiments/runs/h7c_initial"
    speed_multiplier: float = 2.0
    output_phases: int = 29
    cycles: int = 20
    seeds: tuple[int, ...] = (7, 19, 31)
    device: str = "cpu"


def ecological_stability_acceptance(full_acceptance):
    """Exclude only causal gates that require a non-vacuous control error."""
    excluded = {
        "nonvacuous_backbone", "position_causal", "compression_causal",
        "position_error_reduction", "compression_error_reduction", "pass",
    }
    required = {
        name: value for name, value in full_acceptance.items()
        if name not in excluded
    }
    return {**required, "pass": all(required.values())}


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def run_h7d(output_directory, config=None):
    """Evaluate untouched and 2x Kimodo with frozen H7C checkpoints."""
    config = config or H7DConfig()
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    base_config = H7Config(
        seeds=config.seeds,
        cycles=config.cycles,
        device=config.device,
    )
    teacher_config = h7c_teacher_config()
    training_config = h7c_training_config()
    device, volume, graph, _ = _common(base_config)
    _, kimodo_lbs, _ = load_kimodo_cycle(
        base_config.kimodo_motion_path, device
    )
    expected_phases = round(kimodo_lbs.shape[0] / config.speed_multiplier)
    if config.output_phases != expected_phases:
        raise ValueError(
            "output_phases does not match the frozen speed multiplier"
        )
    stress_lbs = periodic_catmull_rom(kimodo_lbs, config.output_phases)
    backbone, provenance = _load_backbone(base_config, device)
    trajectory = simulate_density_teacher_from_lbs(
        stress_lbs, volume, graph, config=teacher_config
    )
    teacher = measure_density_teacher(trajectory, volume, graph)
    diagnostic = sample_density_states(
        trajectory, volume, 300000, 8101, teacher_config
    )
    backbone_metrics = None
    seed_reports = []
    for seed in config.seeds:
        hybrid = _load_checkpoint(
            Path(config.checkpoint_directory),
            seed,
            base_config,
            training_config,
            backbone,
            device,
        )
        training = json.loads((
            Path(config.checkpoint_directory) / f"seed{seed}" / "training.json"
        ).read_text())
        one_step = density_acceleration_nrmse(
            hybrid.density_residual, diagnostic
        )
        if backbone_metrics is None:
            control = rollout_hybrid_density(
                hybrid, trajectory, volume, graph, teacher_config,
                cycles=config.cycles, density_enabled=False,
            )
            backbone_metrics = measure_h7_rollout(
                control, trajectory, volume, graph, teacher_config
            )
            del control
        result = rollout_hybrid_density(
            hybrid, trajectory, volume, graph, teacher_config,
            cycles=config.cycles, density_enabled=True,
        )
        learned = measure_h7_rollout(
            result, trajectory, volume, graph, teacher_config
        )
        gates = acceptance_h7(
            teacher,
            {"acceleration_nrmse": training["acceleration_nrmse"]},
            learned,
            backbone_metrics,
        )
        seed_reports.append({
            "seed": int(seed),
            "training": training,
            "stress_one_step_diagnostic": one_step,
            "hybrid_rollout": learned,
            "acceptance": gates,
        })
        torch.save(
            result.residual[-1].cpu(),
            output / f"kimodo_2x_seed{seed}_render_state.pt",
        )
        print(
            f"H7D Kimodo 2x seed {seed}: "
            f"{'PASS' if gates['pass'] else 'FAIL'} "
            f"hybrid={learned['position_rms'] * 1000:.3f}mm "
            f"backbone={backbone_metrics['position_rms'] * 1000:.3f}mm",
            flush=True,
        )
        del result, hybrid
        if device.type == "cuda":
            torch.cuda.empty_cache()

    h7c = json.loads((
        Path(config.checkpoint_directory) / "metrics.json"
    ).read_text())
    untouched = next(
        item for item in h7c["motions"]
        if item["name"] == "kimodo_weight_shift"
    )
    ecological = [
        {
            "seed": item["seed"],
            "acceptance": ecological_stability_acceptance(item["acceptance"]),
        }
        for item in untouched["seeds"]
    ]
    ecological_pass = all(item["acceptance"]["pass"] for item in ecological)
    stress_pass = all(item["acceptance"]["pass"] for item in seed_reports)
    report = {
        "experiment": "H7D",
        "description": "frozen Kimodo excitation audit",
        "config": asdict(config),
        "teacher_config": asdict(teacher_config),
        "training_config": asdict(training_config),
        "backbone": provenance,
        "untouched_kimodo": {
            "source": str(Path(config.checkpoint_directory) / "metrics.json"),
            "backbone_rollout": untouched["backbone_rollout"],
            "seeds": ecological,
            "ecological_stability_pass": ecological_pass,
        },
        "kimodo_2x": {
            "phase_count": int(stress_lbs.shape[0]),
            "teacher": teacher,
            "backbone_rollout": backbone_metrics,
            "seeds": seed_reports,
            "all_seeds_pass": stress_pass,
        },
        "acceptance": {
            "untouched_ecological_stability": ecological_pass,
            "kimodo_2x_full_causal": stress_pass,
            "pass": ecological_pass and stress_pass,
        },
        "elapsed_seconds": time.time() - started,
    }
    _write_json(output / "metrics.json", report)
    (output / "RUN.md").write_text(
        f"# H7D run: {'PASS' if report['acceptance']['pass'] else 'FAIL'}\n\n"
        f"- Untouched ecological stability: `{ecological_pass}`\n"
        f"- Kimodo 2x full causal gates: `{stress_pass}`\n"
    )
    return report
