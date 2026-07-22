"""Reproducible H1 continuous-deficit recruitment experiment."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

import torch

from .deficit_dynamics import DeficitDynamics, DeficitDynamicsConfig
from .dynamics import ParticleNCARule
from .experiment import _device
from .metrics import acceptance_h1, measure_state
from .morphology import build_h_body_plan, deform_body_plan
from .particles import ParticleSystem
from .render import render_frame, save_contact_sheet, save_gif
from .skeleton import HScaffold


@dataclass(frozen=True)
class H1Config:
    seed: int = 7
    steps: int = 600
    motion_start: int = 400
    motion_rate: float = 0.055
    feed_per_step: int = 4
    capture_every: int = 8
    capacity_extra: int = 48
    spacing: float = 0.14
    tissue_radius: float = 0.24
    splat_radius_scale: float = 0.20
    image_size: int = 320
    device: str = "cpu"
    pressure_enabled: bool = True
    recruitment: str = "deficit"
    plastic_material_enabled: bool = True


def run_h1(output_directory, config=None):
    """Run one H1/control arm and write a complete evidence bundle."""
    config = config or H1Config()
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
    dynamics = DeficitDynamics(
        body_plan, DeficitDynamicsConfig(),
        pressure_enabled=config.pressure_enabled,
        recruitment=config.recruitment,
        plastic_material_enabled=config.plastic_material_enabled,
    )
    learned_rule = ParticleNCARule().to(device).eval()
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    frames, capture_steps, timeline = [], [], []
    assembly_metrics = None
    started = time.time()
    with torch.no_grad():
        for step in range(config.steps):
            moving = step >= config.motion_start
            if step == config.motion_start:
                particles.lock_material()
            motion_time = 0.0 if not moving else (
                step - config.motion_start + 1
            ) * config.motion_rate
            skeleton_frame = scaffold.frame(motion_time, device=device)
            target_positions, gap_positions = deform_body_plan(
                body_plan, skeleton_frame
            )
            remaining = body_plan.site_count - particles.active_count
            particles.feed_unassigned(
                min(config.feed_per_step, max(remaining, 0)), generator
            )
            diagnostics = dynamics.step(
                particles, target_positions, skeleton_frame, motion_time,
                moving, learned_rule,
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
                    splat_radius_world=config.splat_radius_scale * body_plan.spacing,
                    label=(
                        f"{config.recruitment} step {step:03d} "
                        f"cells {particles.active_count:03d}"
                    ),
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
    gates = acceptance_h1(assembly_metrics, moving_metrics, body_plan.site_count)
    report = {
        "experiment": "H1",
        "description": "continuous tissue-deficit particle recruitment",
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
    (output_directory / "RUN.md").write_text(f"""# H1 run: {verdict}

- Seed: `{config.seed}`
- Recruitment: `{config.recruitment}`
- Pressure enabled: `{config.pressure_enabled}`
- Plastic material enabled: `{config.plastic_material_enabled}`
- Sites / capacity: `{body_plan.site_count}` / `{particles.capacity}`
- Assembly coverage: `{assembly_metrics['coverage']:.4f}`
- Moving coverage: `{moving_metrics['coverage']:.4f}`
- Moving tracking error: `{moving_metrics['mean_tracking_error']:.4f}`
- Density error: `{moving_metrics['density_relative_error']:.4f}`
- Committed cells: `{moving_metrics['committed']}`
- Locked material cells: `{moving_metrics['material_locked']}`
- Checker field accuracy: `{moving_metrics['checker_field_accuracy']:.4f}`
- Exposed assignments: `{moving_metrics['exposed_assignments']}`
- Crowding: `{moving_metrics['crowding_fraction']:.4f}`
- Gap occupancy: `{moving_metrics['gap_occupancy']}`
- Acceptance: `{json.dumps(gates, sort_keys=True)}`
""")
    return report
