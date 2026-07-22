"""Reproducible H2 nonuniform morphogenesis and wound-repair experiment."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

import torch

from .deficit_dynamics import DeficitDynamics, DeficitDynamicsConfig
from .dynamics import ParticleNCARule
from .experiment import _device
from .h2_metrics import acceptance_h2, measure_h2
from .h2_morphology import build_h2_body_plan
from .morphology import deform_body_plan
from .particles import ParticleSystem
from .render import render_frame, save_contact_sheet, save_gif
from .skeleton import HScaffold


@dataclass(frozen=True)
class H2Config:
    seed: int = 7
    steps: int = 820
    assembly_lock_step: int = 350
    damage_step: int = 400
    motion_start: int = 650
    motion_rate: float = 0.055
    feed_per_step: int = 4
    capture_every: int = 10
    capacity_extra: int = 72
    spacing: float = 0.14
    splat_radius_scale: float = 0.20
    image_size: int = 320
    device: str = "cpu"
    pressure_enabled: bool = True
    recruitment: str = "hierarchical_deficit"
    plastic_material_enabled: bool = True
    nearest_bone_uses_target_density: bool = True
    local_maturation_enabled: bool = True
    locked_deficit_blend: float = 0.0
    deficit_log_weight: float = 2.2


def _remove_wound_cells(particles, body_plan, target_positions):
    indices, positions, _, _ = particles.active_tensors()
    nearest_site = torch.cdist(positions, target_positions).argmin(dim=1)
    selected = indices[body_plan.wound_mask[nearest_site]]
    return particles.remove(selected)


def run_h2(output_directory, config=None):
    """Run one H2/control arm and write its complete evidence bundle."""
    config = config or H2Config()
    if not (
        0 < config.assembly_lock_step < config.damage_step
        < config.motion_start < config.steps
    ):
        raise ValueError("H2 phase steps must be strictly ordered")
    device = _device(config.device)
    torch.manual_seed(config.seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)
    scaffold = HScaffold()
    body_plan = build_h2_body_plan(
        scaffold, spacing=config.spacing, device=device
    )
    particles = ParticleSystem(
        body_plan.site_count + config.capacity_extra, device=device
    )
    dynamics = DeficitDynamics(
        body_plan,
        DeficitDynamicsConfig(
            moving_deficit_blend=config.locked_deficit_blend,
            deficit_log_weight=config.deficit_log_weight,
        ),
        pressure_enabled=config.pressure_enabled,
        recruitment=config.recruitment,
        plastic_material_enabled=config.plastic_material_enabled,
        nearest_bone_uses_target_density=(
            config.nearest_bone_uses_target_density
        ),
    )
    learned_rule = ParticleNCARule().to(device).eval()
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    frames, capture_steps, timeline = [], [], []
    pre_wound = damaged = repaired = None
    removed_count = 0
    survivor_indices = None
    survivor_positions = None
    started = time.time()
    with torch.no_grad():
        for step in range(config.steps):
            moving = step >= config.motion_start
            motion_time = 0.0 if not moving else (
                step - config.motion_start + 1
            ) * config.motion_rate
            frame = scaffold.frame(motion_time, device=device)
            targets, gaps = deform_body_plan(body_plan, frame)

            if step < config.damage_step:
                needed = body_plan.site_count - particles.active_count
                particles.feed_unassigned(
                    min(config.feed_per_step, max(needed, 0)), generator,
                    generation=0,
                )
            elif step > config.damage_step and step < config.motion_start:
                needed = body_plan.site_count - particles.active_count
                particles.feed_unassigned(
                    min(config.feed_per_step, max(needed, 0)), generator,
                    generation=1,
                )

            if step == config.assembly_lock_step:
                particles.lock_material()
            if step == config.damage_step:
                particles.lock_material()
                pre_wound = measure_h2(
                    particles, body_plan, targets, gaps, frame
                )
                survivor_indices = torch.nonzero(particles.active).flatten()
                survivor_positions = particles.positions[survivor_indices].clone()
                removed_count = _remove_wound_cells(
                    particles, body_plan, targets
                )
                survivor_mask = particles.active[survivor_indices]
                survivor_indices = survivor_indices[survivor_mask]
                survivor_positions = survivor_positions[survivor_mask]
                damaged = measure_h2(
                    particles, body_plan, targets, gaps, frame
                )

            allow_maturation = (
                config.local_maturation_enabled
                and config.damage_step < step < config.motion_start
            )
            diagnostics = dynamics.step(
                particles, targets, frame, motion_time, moving, learned_rule,
                allow_local_maturation=allow_maturation,
            )
            if step == config.motion_start - 1:
                repaired = measure_h2(
                    particles, body_plan, targets, gaps, frame
                )

            if step % 20 == 0 or step in {
                config.damage_step, config.motion_start - 1, config.steps - 1
            }:
                metrics = measure_h2(
                    particles, body_plan, targets, gaps, frame
                )
                timeline.append({
                    "step": step,
                    "motion_time": motion_time,
                    **diagnostics,
                    **metrics,
                })
            if step % config.capture_every == 0 or step in {
                config.damage_step, config.motion_start - 1, config.steps - 1
            }:
                phase = (
                    "grow" if step < config.damage_step
                    else "repair" if step < config.motion_start
                    else "move"
                )
                frames.append(render_frame(
                    particles, frame, size=config.image_size,
                    splat_radius_world=(
                        config.splat_radius_scale * body_plan.spacing
                    ),
                    label=(
                        f"{config.recruitment} {phase} {step:03d} "
                        f"cells {particles.active_count:03d}"
                    ),
                ))
                capture_steps.append(step)

    final_time = (config.steps - config.motion_start) * config.motion_rate
    final_frame = scaffold.frame(final_time, device=device)
    final_targets, final_gaps = deform_body_plan(body_plan, final_frame)
    moving = measure_h2(
        particles, body_plan, final_targets, final_gaps, final_frame
    )
    if any(state is None for state in (pre_wound, damaged, repaired)):
        raise RuntimeError("H2 did not capture every required phase")
    healthy_displacement = float((
        particles.positions[survivor_indices] - survivor_positions
    ).norm(dim=1).mean().item())
    gates = acceptance_h2(
        pre_wound, damaged, repaired, moving, body_plan.site_count
    )
    report = {
        "experiment": "H2",
        "description": "nonuniform morphogenesis and wound repair",
        "config": asdict(config),
        "dynamics": asdict(dynamics.config),
        "region_names": body_plan.region_names,
        "region_site_counts": torch.bincount(
            body_plan.region, minlength=len(body_plan.region_names)
        ).tolist(),
        "site_count": body_plan.site_count,
        "wound_site_count": int(body_plan.wound_mask.sum().item()),
        "removed_count": removed_count,
        "healthy_mean_displacement": healthy_displacement,
        "pre_wound": pre_wound,
        "damaged": damaged,
        "repaired": repaired,
        "moving": moving,
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
        0,
        feed_steps,
        config.assembly_lock_step,
        config.damage_step,
        min(config.motion_start - 1, config.damage_step + 50),
        config.motion_start - 1,
        min(config.steps - 1, config.motion_start + 48),
        config.steps - 1,
    ]
    milestone_indices = sorted(set(
        min(range(len(capture_steps)), key=lambda index: abs(
            capture_steps[index] - desired
        ))
        for desired in desired_steps
    ))
    save_contact_sheet(
        [frames[index] for index in milestone_indices],
        output_directory / "contact_sheet.png", columns=4,
    )
    verdict = "PASS" if gates["pass"] else "FAIL"
    (output_directory / "RUN.md").write_text(f"""# H2 run: {verdict}

- Seed: `{config.seed}`
- Recruitment: `{config.recruitment}`
- Pressure enabled: `{config.pressure_enabled}`
- Plastic material enabled: `{config.plastic_material_enabled}`
- Sites / wound / removed: `{body_plan.site_count}` / `{int(body_plan.wound_mask.sum())}` / `{removed_count}`
- Pre-wound coverage: `{pre_wound['coverage']:.4f}`
- Damaged wound coverage: `{damaged['wound_coverage']:.4f}`
- Repaired wound coverage: `{repaired['wound_coverage']:.4f}`
- Repair localization: `{repaired['repair_wound_localization']:.4f}`
- Repair lock fraction: `{repaired['repair_lock_fraction']:.4f}`
- Coarse guided cells: `{moving['guided_cells']}`
- Moving coverage: `{moving['coverage']:.4f}`
- Moving checker field: `{moving['checker_field_accuracy']:.4f}`
- Healthy displacement: `{healthy_displacement:.4f}`
- Acceptance: `{json.dumps(gates, sort_keys=True)}`
""")
    return report
