#!/usr/bin/env python3
"""Render a radius/opacity sweep for one validated rigged volume frame."""

import argparse
from pathlib import Path

from PIL import ImageDraw
import torch

from flesh_and_bone.h4_render import (
    load_base_color,
    render_colored_splats,
    sample_texture,
)
from flesh_and_bone.h4_volume import load_h4_volume
from flesh_and_bone.render import save_contact_sheet
from flesh_and_bone.rig_asset import linear_skin, load_rig_asset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rig", default="model/derived/meshy_blonde_h4_rig.npz"
    )
    parser.add_argument(
        "--volume",
        default="model/derived/meshy_blonde_h4_volume_p0175.npz",
    )
    parser.add_argument(
        "--archive", default="model/Meshy_AI_Blonde_female_mechani_biped.zip"
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("experiments/runs/splat_sweep_p0175"),
    )
    parser.add_argument("--size", type=int, default=480)
    args = parser.parse_args()

    asset = load_rig_asset(args.rig, dtype=torch.float32)
    volume = load_h4_volume(args.volume, dtype=torch.float32)
    positions = linear_skin(
        volume.points, volume.weights, asset.skin_matrices[:1]
    )[0].cpu().numpy()
    colors = sample_texture(
        load_base_color(args.archive), volume.uv.cpu().numpy()
    )
    bones = asset.bone_endpoints[0].cpu().numpy()
    radius_scales = (0.30, 0.40, 0.50, 0.60)
    opacities = (0.52, 0.72, 0.90)
    frames = []
    faces = []
    for opacity in opacities:
        for radius_scale in radius_scales:
            label = f"radius {radius_scale:.2f} pitch opacity {opacity:.2f}"
            frame = render_colored_splats(
                positions,
                colors,
                bones,
                splat_radius=radius_scale * volume.metadata["pitch"],
                splat_scale=volume.splat_scale.cpu().numpy(),
                size=args.size,
                opacity=opacity,
                label=label,
            )
            frames.append(frame)
            left = round(0.28 * args.size)
            top = 0
            right = round(0.72 * args.size)
            bottom = round(0.40 * args.size)
            face = frame.crop((left, top, right, bottom)).resize(
                (args.size, args.size)
            )
            draw = ImageDraw.Draw(face)
            draw.rectangle((5, 5, 8 + 7 * len(label), 23), fill=(6, 8, 11))
            draw.text((9, 8), label, fill=(235, 238, 240))
            faces.append(face)
    args.output.mkdir(parents=True, exist_ok=True)
    save_contact_sheet(frames, args.output / "full_body_sweep.png", columns=4)
    save_contact_sheet(faces, args.output / "face_sweep.png", columns=4)


if __name__ == "__main__":
    main()
