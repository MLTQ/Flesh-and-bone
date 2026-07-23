#!/usr/bin/env python3
"""Render H7C/H7D density-blind versus bounded-hybrid comparisons."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from flesh_and_bone.density_rollout import rollout_hybrid_density
from flesh_and_bone.density_teacher import simulate_density_teacher_from_lbs
from flesh_and_bone.h4_render import (
    error_colors,
    load_base_color,
    render_colored_splats,
    sample_texture,
)
from flesh_and_bone.h6k_experiment import load_kimodo_cycle
from flesh_and_bone.h7_experiment import (
    H7Config,
    _common,
    _load_backbone,
    _load_checkpoint,
    h7c_teacher_config,
    h7c_training_config,
)
from flesh_and_bone.motion_variants import periodic_catmull_rom
from flesh_and_bone.render import save_contact_sheet, save_gif


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path("experiments/runs/h7_visuals"),
    )
    parser.add_argument(
        "--h7c", type=Path, default=Path("experiments/runs/h7c_initial")
    )
    parser.add_argument(
        "--h7d", type=Path,
        default=Path("experiments/runs/h7d_frozen_stress"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--size", type=int, default=420)
    parser.add_argument("--cell-stride", type=int, default=3)
    return parser.parse_args()


def _paired(left, right):
    output = Image.new(
        "RGB", (left.width + right.width, left.height), (6, 8, 10)
    )
    output.paste(left, (0, 0))
    output.paste(right, (left.width, 0))
    return output


def _render_motion(name, lbs, bones, teacher, backbone, hybrid, colors,
                   volume, selection, output, size):
    texture_frames = []
    error_frames = []
    scale = volume.splat_scale.detach().cpu().numpy()[selection]
    radius = 0.65 * float(volume.metadata["pitch"])
    lbs = lbs.detach().cpu()
    bones = bones.detach().cpu()
    teacher = teacher.detach().cpu()
    backbone = backbone.detach().cpu()
    hybrid = hybrid.detach().cpu()
    backbone_error = (backbone - teacher).norm(dim=-1)
    ceiling = float(torch.quantile(backbone_error.reshape(-1), 0.99).item())
    for phase in range(lbs.shape[0]):
        common = {
            "bone_endpoints": bones[phase].numpy(),
            "splat_radius": radius,
            "splat_scale": scale,
            "size": size,
            "opacity": 0.80,
        }
        backbone_points = (lbs[phase] + backbone[phase])[selection].numpy()
        hybrid_points = (lbs[phase] + hybrid[phase])[selection].numpy()
        left = render_colored_splats(
            backbone_points,
            colors,
            label=f"{name} density-blind phase {phase:02d}",
            **common,
        )
        right = render_colored_splats(
            hybrid_points,
            colors,
            label=f"{name} bounded hybrid phase {phase:02d}",
            **common,
        )
        texture_frames.append(_paired(left, right))
        left_error = render_colored_splats(
            backbone_points,
            error_colors(backbone_error[phase, selection].numpy(), ceiling),
            label=f"density-blind error (red={ceiling * 1000:.2f}mm)",
            **common,
        )
        hybrid_error = (hybrid[phase] - teacher[phase]).norm(dim=-1)
        right_error = render_colored_splats(
            hybrid_points,
            error_colors(hybrid_error[selection].numpy(), ceiling),
            label=f"bounded hybrid error (same scale)",
            **common,
        )
        error_frames.append(_paired(left_error, right_error))
        if phase % 5 == 0:
            print(f"H7 visual {name}: {phase + 1}/{lbs.shape[0]}", flush=True)

    motion_output = output / name
    motion_output.mkdir(parents=True, exist_ok=True)
    save_gif(texture_frames, motion_output / "backbone_vs_hybrid.gif", 85)
    save_gif(error_frames, motion_output / "error_heatmap.gif", 85)
    milestones = torch.linspace(
        0, lbs.shape[0] - 1, 5
    ).round().to(torch.long).unique().tolist()
    save_contact_sheet(
        [texture_frames[index] for index in milestones],
        motion_output / "backbone_vs_hybrid_contact_sheet.png",
        columns=1,
    )
    save_contact_sheet(
        [error_frames[index] for index in milestones],
        motion_output / "error_heatmap_contact_sheet.png",
        columns=1,
    )


def main():
    args = _arguments()
    config = H7Config(device=args.device, cycles=20)
    teacher_config = h7c_teacher_config()
    training_config = h7c_training_config()
    device, volume, graph, motions = _common(config)
    backbone_rule, _ = _load_backbone(config, device)
    hybrid_rule = _load_checkpoint(
        args.h7c,
        args.seed,
        config,
        training_config,
        backbone_rule,
        device,
    )
    texture = load_base_color(
        "model/Meshy_AI_Blonde_female_mechani_biped.zip"
    )
    all_colors = sample_texture(texture, volume.uv.detach().cpu().numpy())
    selection = np.arange(0, volume.cell_count, args.cell_stride)
    colors = all_colors[selection]

    fast = motions["fast_1p526"]
    _, kimodo_lbs, kimodo_bones = load_kimodo_cycle(
        config.kimodo_motion_path, device
    )
    visual_motions = (
        (
            "fast_1p526",
            fast.lbs_positions,
            fast.bone_endpoints,
            args.h7c / f"fast_1p526_seed{args.seed}_render_state.pt",
        ),
        (
            "kimodo_2x",
            periodic_catmull_rom(kimodo_lbs, 29),
            periodic_catmull_rom(kimodo_bones, 29),
            args.h7d / f"kimodo_2x_seed{args.seed}_render_state.pt",
        ),
    )
    for name, lbs, bones, hybrid_path in visual_motions:
        print(f"H7 visual mechanics {name}", flush=True)
        trajectory = simulate_density_teacher_from_lbs(
            lbs, volume, graph, config=teacher_config
        )
        control = rollout_hybrid_density(
            hybrid_rule,
            trajectory,
            volume,
            graph,
            teacher_config,
            cycles=20,
            density_enabled=False,
        )
        hybrid = torch.load(
            hybrid_path, map_location="cpu", weights_only=True
        )
        _render_motion(
            name,
            lbs,
            bones,
            trajectory.residual[:, 0],
            control.residual[-1],
            hybrid,
            colors,
            volume,
            selection,
            args.output,
            args.size,
        )
        del trajectory, control, hybrid
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
