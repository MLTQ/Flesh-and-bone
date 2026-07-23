#!/usr/bin/env python3
"""Render H8 LBS-control versus frozen-hybrid streaming comparisons."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from flesh_and_bone.h4_render import load_base_color, render_colored_splats, sample_texture
from flesh_and_bone.h4_volume import load_h4_volume
from flesh_and_bone.kimodo_diagnostics import hierarchy_segments
from flesh_and_bone.render import save_contact_sheet, save_gif
from flesh_and_bone.rig_asset import load_rig_asset


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=Path("experiments/runs/h8_streaming")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/runs/h8_visuals")
    )
    parser.add_argument("--size", type=int, default=420)
    return parser.parse_args()


def _paired(left, right):
    image = Image.new("RGB", (left.width + right.width, left.height), (6, 8, 10))
    image.paste(left, (0, 0))
    image.paste(right, (left.width, 0))
    return image


def _render_state(path, output, volume, texture_colors, parents, size):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") != "flesh-and-bone-h8-render-v1":
        raise ValueError(f"unsupported H8 render state {path}")
    selection = state["selection"].numpy()
    colors = texture_colors[selection]
    scales = volume.splat_scale.numpy()[selection]
    radius = 0.65 * float(volume.metadata["pitch"])
    lbs = state["lbs"].numpy()
    hybrid = state["hybrid_residual"].numpy()
    bones = state["bones"].numpy()
    segments = hierarchy_segments(bones, parents, include_count=22)
    frames = []
    for index, source_frame in enumerate(state["frame_indices"].tolist()):
        follow = np.array([bones[index, 0, 0, 0], 0.0, bones[index, 0, 0, 2]])
        common = {
            "bone_endpoints": segments[index] - follow,
            "splat_radius": radius,
            "splat_scale": scales,
            "size": int(size),
            "opacity": 0.84,
        }
        left = render_colored_splats(
            lbs[index] - follow,
            colors,
            label=f"rig-only LBS · physics OFF · frame {source_frame + 1}",
            **common,
        )
        right = render_colored_splats(
            lbs[index] + hybrid[index] - follow,
            colors,
            label=f"frozen H7C hybrid · physics ON · frame {source_frame + 1}",
            **common,
        )
        frames.append(_paired(left, right))
    destination = output / path.name.removesuffix("_render_state.pt")
    destination.mkdir(parents=True, exist_ok=True)
    duration_ms = max(
        20,
        round(1000 * int(state["source_frame_count"]) / (30 * len(frames))),
    )
    save_gif(frames, destination / "lbs_vs_hybrid.gif", duration_ms)
    milestones = np.linspace(0, len(frames) - 1, min(8, len(frames))).round().astype(int)
    save_contact_sheet(
        [frames[index] for index in np.unique(milestones)],
        destination / "lbs_vs_hybrid_contact_sheet.png",
        columns=2,
    )
    return destination


def main():
    args = _arguments()
    paths = sorted(args.input.glob("*_render_state.pt"))
    if not paths:
        raise FileNotFoundError(f"no H8 render states in {args.input}")
    volume = load_h4_volume(
        "model/derived/meshy_blonde_h4_volume_p0125.npz", dtype=torch.float32
    )
    asset = load_rig_asset(
        "model/derived/meshy_blonde_h4_rig.npz", dtype=torch.float32
    )
    parents = asset.bone_parents.numpy()
    texture = load_base_color("model/Meshy_AI_Blonde_female_mechani_biped.zip")
    colors = sample_texture(texture, volume.uv.numpy())
    for path in paths:
        destination = _render_state(
            path, args.output, volume, colors, parents, args.size
        )
        print(f"H8 visual: {destination}", flush=True)


if __name__ == "__main__":
    main()
