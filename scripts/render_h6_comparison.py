#!/usr/bin/env python3
"""Render matched full-cycle GIFs for H6M failure and H6C success."""

import argparse
from pathlib import Path

from PIL import Image
import torch

from flesh_and_bone.constitutive_rule import fit_constitutive_rule
from flesh_and_bone.flesh_rollout import rollout_flesh_rule
from flesh_and_bone.flesh_rule import FleshResidualRule
from flesh_and_bone.flesh_teacher import (
    ElasticTeacherConfig,
    build_voxel_graph,
    simulate_teacher,
    simulate_teacher_from_lbs,
    volume_lbs_cycle,
)
from flesh_and_bone.h4_render import (
    load_base_color,
    render_colored_splats,
    sample_texture,
)
from flesh_and_bone.h4_volume import load_h4_volume
from flesh_and_bone.motion_variants import controlled_motion_cycles
from flesh_and_bone.render import save_gif
from flesh_and_bone.rig_asset import load_rig_asset


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/runs/h6_visuals"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--duration-ms", type=int, default=85)
    return parser.parse_args()


def _paired_frames(left, right):
    frames = []
    for first, second in zip(left, right):
        combined = Image.new(
            "RGB", (first.width + second.width, first.height), (6, 8, 10)
        )
        combined.paste(first, (0, 0))
        combined.paste(second, (first.width, 0))
        frames.append(combined)
    return frames


def main():
    args = _arguments()
    device = torch.device(args.device)
    teacher_config = ElasticTeacherConfig(neighbor_coupling=1200.0)
    asset = load_rig_asset(
        "model/derived/meshy_blonde_h4_rig.npz",
        device=device,
        dtype=torch.float32,
    )
    volume = load_h4_volume(
        "model/derived/meshy_blonde_h4_volume_p0125.npz",
        device=device,
        dtype=torch.float32,
    )
    graph = build_voxel_graph(volume.points, volume.metadata["pitch"])
    base_lbs, _ = volume_lbs_cycle(asset, volume)
    fast = next(
        motion
        for motion in controlled_motion_cycles(
            base_lbs, asset.bone_endpoints[:-1]
        )
        if motion.name == "fast_1p526"
    )
    fast_teacher = simulate_teacher_from_lbs(
        fast.lbs_positions, volume, graph, config=teacher_config
    )
    print("rolling out failed H5U MLP", flush=True)
    failed_rule = FleshResidualRule(96).to(device)
    checkpoint = (
        Path("experiments/runs/h5u_final")
        / f"seed{args.seed}"
        / "flesh_rule.pt"
    )
    failed_rule.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    failed_rule.eval()
    failed_rollout, _, _ = rollout_flesh_rule(
        failed_rule,
        fast_teacher,
        volume,
        graph,
        teacher_config,
        cycles=args.cycles,
        neighbor_enabled=True,
    )
    failed = failed_rollout[-1].detach().cpu()
    del failed_rollout, failed_rule
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("identifying and rolling out successful constitutive rule", flush=True)
    replay_teacher = simulate_teacher(
        asset, volume, graph, config=teacher_config
    )
    successful_rule, _ = fit_constitutive_rule(replay_teacher, volume)
    successful_rollout, _, _ = rollout_flesh_rule(
        successful_rule,
        fast_teacher,
        volume,
        graph,
        teacher_config,
        cycles=args.cycles,
        neighbor_enabled=True,
    )
    successful = successful_rollout[-1].detach().cpu()
    del successful_rollout, successful_rule, replay_teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()

    texture = load_base_color(
        "model/Meshy_AI_Blonde_female_mechani_biped.zip"
    )
    colors = sample_texture(texture, volume.uv.detach().cpu().numpy())
    lbs = fast.lbs_positions.detach().cpu()
    bones = fast.bone_endpoints.detach().cpu()
    splat_scale = volume.splat_scale.detach().cpu().numpy()
    radius = 0.50 * volume.metadata["pitch"]
    failed_frames = []
    successful_frames = []
    for phase in range(fast.lbs_positions.shape[0]):
        common = {
            "colors": colors,
            "bone_endpoints": bones[phase].numpy(),
            "splat_radius": radius,
            "splat_scale": splat_scale,
            "size": 720,
            "opacity": 0.72,
        }
        failed_frames.append(render_colored_splats(
            (lbs[phase] + failed[phase]).numpy(),
            label=(
                f"FAIL: MLP seed {args.seed}, cycle {args.cycles}, "
                f"phase {phase:02d}"
            ),
            **common,
        ))
        successful_frames.append(render_colored_splats(
            (lbs[phase] + successful[phase]).numpy(),
            label=(
                f"PASS: constitutive, cycle {args.cycles}, phase {phase:02d}"
            ),
            **common,
        ))
        if phase % 4 == 0:
            print(f"rendered phase {phase + 1}/{lbs.shape[0]}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    save_gif(
        failed_frames,
        args.output / "fast_mlp_failure.gif",
        duration_ms=args.duration_ms,
    )
    save_gif(
        successful_frames,
        args.output / "fast_constitutive_success.gif",
        duration_ms=args.duration_ms,
    )
    save_gif(
        _paired_frames(failed_frames, successful_frames),
        args.output / "fast_failure_vs_success.gif",
        duration_ms=args.duration_ms,
    )
    print(f"wrote animations to {args.output}", flush=True)


if __name__ == "__main__":
    main()
