"""Fine humanoid assembly, learned fate, cheek repair, and motion experiment."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time

import torch

from .deficit_dynamics import DeficitDynamics, DeficitDynamicsConfig
from .dynamics import ParticleNCARule
from .experiment import _device
from .fate_model import LearnedFateSelector, train_fate_model
from .h3_metrics import (
    acceptance_h3_learned,
    acceptance_h3_oracle,
    measure_h3,
    representation_gates,
)
from .h3_morphology import build_h3_body_plan
from .humanoid_skeleton import HumanoidScaffold
from .morphology import deform_body_plan
from .particles import ParticleSystem
from .render import render_frame, save_contact_sheet, save_gif


H3_ARMS = ("oracle", "learned", "local_deficit", "no_shortage")


@dataclass(frozen=True)
class H3Config:
    arm: str = "learned"
    seed: int = 7
    steps: int = 760
    assembly_lock_step: int = 300
    damage_step: int = 340
    motion_start: int = 570
    motion_rate: float = 0.050
    feed_per_step: int = 12
    capture_every: int = 20
    capacity_extra: int = 128
    spacing: float = 0.10
    splat_radius_scale: float = 0.15
    image_size: int = 480
    world_extent: float = 4.1
    device: str = "cpu"
    feed_source: tuple[float, float, float] = (0.0, -2.05, 0.0)
    fate_train_steps: int = 700
    fate_batch_size: int = 512
    deficit_log_weight: float = 2.8
    oracle_pre_coverage: float | None = None


def _remove_wound_cells(particles, body_plan, target_positions):
    indices, positions, _, _ = particles.active_tensors()
    nearest_site = torch.cdist(positions, target_positions).argmin(dim=1)
    selected = indices[body_plan.wound_mask[nearest_site]]
    return particles.remove(selected)


def _arm_components(config, body_plan, device):
    if config.arm not in H3_ARMS:
        raise ValueError(f"arm must be one of {H3_ARMS}")
    model = training_report = selector = None
    recruitment = "deficit" if config.arm == "local_deficit" else (
        "hierarchical_deficit"
    )
    if config.arm in {"learned", "no_shortage"}:
        model, training_report = train_fate_model(
            len(body_plan.region_names),
            seed=config.seed,
            device=device,
            steps=config.fate_train_steps,
            batch_size=config.fate_batch_size,
        )
        selector = LearnedFateSelector(
            model,
            body_plan.spacing,
            expose_shortage=config.arm == "learned",
        )
    return recruitment, selector, model, training_report


def run_h3(output_directory, config=None):
    """Run one H3 arm and write a complete, self-describing evidence bundle."""
    config = config or H3Config()
    if not (
        0 < config.assembly_lock_step < config.damage_step
        < config.motion_start < config.steps
    ):
        raise ValueError("H3 phase steps must be strictly ordered")
    device = _device(config.device)
    torch.manual_seed(config.seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)
    scaffold = HumanoidScaffold()
    body_plan = build_h3_body_plan(
        scaffold, spacing=config.spacing, device=device
    )
    particles = ParticleSystem(
        body_plan.site_count + config.capacity_extra,
        bone_count=scaffold.bone_count,
        device=device,
    )
    recruitment, selector, fate_model, training_report = _arm_components(
        config, body_plan, device
    )
    dynamics = DeficitDynamics(
        body_plan,
        DeficitDynamicsConfig(
            moving_deficit_blend=0.0,
            deficit_log_weight=config.deficit_log_weight,
        ),
        pressure_enabled=True,
        recruitment=recruitment,
        plastic_material_enabled=True,
        nearest_bone_uses_target_density=True,
        region_selector=selector,
    )
    learned_rule = ParticleNCARule().to(device).eval()
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    representation_values, representation = representation_gates(
        body_plan, scaffold.bone_count, config.splat_radius_scale
    )
    frames, capture_steps, timeline = [], [], []
    pre_wound = damaged = repaired = None
    removed_count = 0
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
                    min(config.feed_per_step, max(needed, 0)),
                    generator,
                    source=config.feed_source,
                    generation=0,
                )
            elif step > config.damage_step and step < config.motion_start:
                needed = body_plan.site_count - particles.active_count
                particles.feed_unassigned(
                    min(config.feed_per_step, max(needed, 0)),
                    generator,
                    source=config.feed_source,
                    generation=1,
                )

            if step == config.assembly_lock_step:
                particles.lock_material()
            if step == config.damage_step:
                particles.lock_material()
                pre_wound = measure_h3(
                    particles, body_plan, targets, gaps, frame
                )
                removed_count = _remove_wound_cells(
                    particles, body_plan, targets
                )
                damaged = measure_h3(
                    particles, body_plan, targets, gaps, frame
                )

            diagnostics = dynamics.step(
                particles,
                targets,
                frame,
                motion_time,
                moving,
                learned_rule,
                allow_local_maturation=(
                    config.damage_step < step < config.motion_start
                ),
            )
            if step == config.motion_start - 1:
                repaired = measure_h3(
                    particles, body_plan, targets, gaps, frame
                )

            if step % 40 == 0 or step in {
                config.damage_step, config.motion_start - 1, config.steps - 1
            }:
                timeline.append({
                    "step": step,
                    "motion_time": motion_time,
                    **diagnostics,
                    **measure_h3(
                        particles, body_plan, targets, gaps, frame
                    ),
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
                    particles,
                    frame,
                    size=config.image_size,
                    splat_radius_world=(
                        config.splat_radius_scale * body_plan.spacing
                    ),
                    world_extent=config.world_extent,
                    minimum_sigma_pixels=1.0,
                    label=(
                        f"{config.arm} {phase} {step:03d} "
                        f"cells {particles.active_count:04d}"
                    ),
                ))
                capture_steps.append(step)

    final_time = (config.steps - config.motion_start) * config.motion_rate
    final_frame = scaffold.frame(final_time, device=device)
    final_targets, final_gaps = deform_body_plan(body_plan, final_frame)
    moving = measure_h3(
        particles, body_plan, final_targets, final_gaps, final_frame
    )
    if any(state is None for state in (pre_wound, damaged, repaired)):
        raise RuntimeError("H3 did not capture every required phase")

    if config.arm == "oracle":
        gates = acceptance_h3_oracle(
            pre_wound, moving, representation
        )
    else:
        agreement = (
            training_report.holdout_agreement
            if training_report is not None else 0.0
        )
        oracle_pre = (
            config.oracle_pre_coverage
            if config.oracle_pre_coverage is not None else 1.0
        )
        gates = acceptance_h3_learned(
            pre_wound,
            damaged,
            repaired,
            moving,
            body_plan.site_count,
            representation,
            agreement,
            oracle_pre,
        )

    report = {
        "experiment": "H3",
        "description": "fine humanoid envelope and learned coarse fate",
        "arm": config.arm,
        "config": asdict(config),
        "dynamics": asdict(dynamics.config),
        "representation": representation_values,
        "representation_acceptance": representation,
        "fate_training": (
            asdict(training_report) if training_report is not None else None
        ),
        "region_names": body_plan.region_names,
        "region_site_counts": torch.bincount(
            body_plan.region, minlength=len(body_plan.region_names)
        ).tolist(),
        "site_count": body_plan.site_count,
        "wound_site_count": int(body_plan.wound_mask.sum().item()),
        "removed_count": removed_count,
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
    if fate_model is not None:
        torch.save(
            {
                key: value.detach().cpu()
                for key, value in fate_model.state_dict().items()
            },
            output_directory / "fate_model.pt",
        )
    save_gif(frames, output_directory / "animation.gif")
    feed_steps = math.ceil(body_plan.site_count / config.feed_per_step)
    desired_steps = [
        0,
        feed_steps,
        config.assembly_lock_step,
        config.damage_step,
        min(config.motion_start - 1, config.damage_step + 60),
        config.motion_start - 1,
        min(config.steps - 1, config.motion_start + 60),
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
        output_directory / "contact_sheet.png",
        columns=4,
    )
    verdict = "PASS" if gates["pass"] else "FAIL"
    holdout = (
        training_report.holdout_agreement
        if training_report is not None else 0.0
    )
    (output_directory / "RUN.md").write_text(f"""# H3 run: {verdict}

- Arm: `{config.arm}`
- Seed: `{config.seed}`
- Bones / regions / sites: `{scaffold.bone_count}` / `{len(body_plan.region_names)}` / `{body_plan.site_count}`
- World splat maximum: `{representation_values['maximum_world_splat_radius']:.5f}`
- Extra-skeletal fraction: `{representation_values['extra_skeletal_fraction']:.4f}`
- Fate holdout agreement: `{holdout:.4f}`
- Pre-wound coverage / minimum region: `{pre_wound['coverage']:.4f}` / `{pre_wound['minimum_region_coverage']:.4f}`
- Damaged / repaired wound: `{damaged['wound_coverage']:.4f}` / `{repaired['wound_coverage']:.4f}`
- Repair localization / lock: `{repaired['repair_wound_localization']:.4f}` / `{repaired['repair_lock_fraction']:.4f}`
- Moving coverage / checker: `{moving['coverage']:.4f}` / `{moving['checker_field_accuracy']:.4f}`
- Acceptance: `{json.dumps(gates, sort_keys=True)}`
""")
    return report
