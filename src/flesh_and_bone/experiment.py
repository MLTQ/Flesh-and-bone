"""Reproducible H0 assembly-and-motion experiment orchestration."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

import torch

from .dynamics import DynamicsConfig, MechanicalDynamics, ParticleNCARule
from .metrics import acceptance, measure_state
from .morphology import build_h_body_plan, deform_body_plan
from .particles import ParticleSystem
from .render import render_frame, save_contact_sheet, save_gif
from .skeleton import HScaffold


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 7
    steps: int = 420
    motion_start: int = 230
    motion_rate: float = 0.060
    feed_per_step: int = 8
    capture_every: int = 6
    capacity_extra: int = 48
    spacing: float = 0.14
    tissue_radius: float = 0.24
    image_size: int = 320
    device: str = "cpu"


def _device(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return torch.device(name)


def run_h0(output_directory, config=None):
    """Run H0, write evidence artifacts, and return the JSON-compatible report."""
    config = config or ExperimentConfig()
    device = _device(config.device)
    torch.manual_seed(config.seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)
    scaffold = HScaffold()
    body_plan = build_h_body_plan(
        scaffold, spacing=config.spacing, radius=config.tissue_radius,
        device=device,
    )
    particles = ParticleSystem(
        body_plan.site_count + config.capacity_extra, device=device
    )
    dynamics = MechanicalDynamics(body_plan, DynamicsConfig())
    learned_rule = ParticleNCARule().to(device).eval()
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    frames = []
    capture_steps = []
    timeline = []
    assembly_metrics = None
    started = time.time()
    with torch.no_grad():
        for step in range(config.steps):
            motion_time = 0.0 if step < config.motion_start else (
                step - config.motion_start + 1
            ) * config.motion_rate
            skeleton_frame = scaffold.frame(motion_time, device=device)
            target_positions, gap_positions = deform_body_plan(
                body_plan, skeleton_frame
            )
            particles.feed(
                body_plan, target_positions, config.feed_per_step, generator
            )
            diagnostics = dynamics.step(
                particles, target_positions, motion_time, learned_rule
            )
            if step == config.motion_start - 1:
                assembly_metrics = measure_state(
                    particles, body_plan, target_positions, gap_positions
                )
            if step % 20 == 0 or step == config.steps - 1:
                metrics = measure_state(
                    particles, body_plan, target_positions, gap_positions
                )
                timeline.append({
                    "step": step,
                    "motion_time": motion_time,
                    **diagnostics,
                    **metrics,
                })
            if step % config.capture_every == 0 or step == config.steps - 1:
                frames.append(render_frame(
                    particles, skeleton_frame, size=config.image_size,
                    splat_radius_world=0.31 * body_plan.spacing,
                    label=f"step {step:03d}  cells {particles.active_count:03d}",
                ))
                capture_steps.append(step)

    final_time = (config.steps - config.motion_start) * config.motion_rate
    final_frame = scaffold.frame(final_time, device=device)
    final_targets, final_gaps = deform_body_plan(body_plan, final_frame)
    moving_metrics = measure_state(
        particles, body_plan, final_targets, final_gaps
    )
    if assembly_metrics is None:
        raise RuntimeError("motion_start must occur within the experiment")
    gates = acceptance(assembly_metrics, moving_metrics, body_plan.site_count)
    report = {
        "experiment": "H0",
        "description": "five-bone H mechanical particle control",
        "config": asdict(config),
        "dynamics": asdict(dynamics.config),
        "site_count": body_plan.site_count,
        "capacity": particles.capacity,
        "assembly": assembly_metrics,
        "moving": moving_metrics,
        "acceptance": gates,
        "timeline": timeline,
        "elapsed_seconds": time.time() - started,
    }
    (output_directory / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    save_gif(frames, output_directory / "animation.gif")
    feed_steps = math.ceil(body_plan.site_count / config.feed_per_step)
    desired_steps = [
        0, feed_steps // 2, feed_steps, config.motion_start - 1,
        min(config.steps - 1, config.motion_start + 48), config.steps - 1,
    ]
    milestone_indices = sorted(set(
        min(range(len(capture_steps)), key=lambda index: abs(
            capture_steps[index] - desired
        ))
        for desired in desired_steps
    ))
    save_contact_sheet(
        [frames[index] for index in milestone_indices],
        output_directory / "contact_sheet.png",
    )
    verdict = "PASS" if gates["pass"] else "FAIL"
    note = f"""# H0 run: {verdict}

- Seed: `{config.seed}`
- Device: `{config.device}`
- Sites / capacity: `{body_plan.site_count}` / `{particles.capacity}`
- Assembly coverage: `{assembly_metrics['coverage']:.4f}`
- Moving coverage: `{moving_metrics['coverage']:.4f}`
- Moving tracking error: `{moving_metrics['mean_tracking_error']:.4f}`
- Committed cells: `{moving_metrics['committed']}`
- Crowding: `{moving_metrics['crowding_fraction']:.4f}`
- Gap occupancy: `{moving_metrics['gap_occupancy']}`
- Acceptance: `{json.dumps(gates, sort_keys=True)}`

This generated note records the resolved run. The repository-level experiment
ledger contains the human interpretation and next decision.
"""
    (output_directory / "RUN.md").write_text(note)
    return report
