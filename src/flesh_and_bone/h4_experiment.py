"""H4 production-rig round-trip and variable-thickness target experiment."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch

from .h4_metrics import acceptance_h4, select_influence_count
from .h4_render import (
    error_colors,
    load_base_color,
    render_colored_splats,
    sample_texture,
)
from .h4_volume import build_h4_volume, save_h4_volume
from .render import save_contact_sheet, save_gif
from .rig_asset import (
    averaged_vertex_uv,
    evaluate_skinning,
    linear_skin,
    load_rig_asset,
    normalized_topk_weights,
)


@dataclass(frozen=True)
class H4Config:
    archive_path: str = "model/Meshy_AI_Blonde_female_mechani_biped.zip"
    rig_asset_path: str = "model/derived/meshy_blonde_h4_rig.npz"
    pitch: float = 0.025
    image_size: int = 480
    surface_splat_radius: float = 0.006
    volume_splat_radius_scale: float = 0.30
    gif_duration_ms: int = 70


def _skin_in_frames(points, weights, skin_matrices):
    return torch.cat([
        linear_skin(points, weights, skin_matrices[index:index + 1])
        for index in range(skin_matrices.shape[0])
    ], dim=0)


def run_h4(output_directory, config=None):
    """Run extraction-independent H4 rig, surface, and volume validation."""
    config = config or H4Config()
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    started = time.time()
    asset = load_rig_asset(config.rig_asset_path, dtype=torch.float64)

    rest_error = (asset.rest_vertices - asset.animated_bind_vertices).norm(dim=1)
    weight_sum_error = (asset.weights.sum(dim=1) - 1).abs()
    loop_matrix_error = asset.skin_matrices[0] - asset.skin_matrices[-1]
    provenance = {
        "source_archive": asset.metadata["source_archive"],
        "source_sha256": asset.metadata["source_sha256"],
        "static_member": asset.metadata["static_member"],
        "animation_member": asset.metadata["animation_member"],
        "source_members_recorded": bool(
            asset.metadata["static_member"]
            and asset.metadata["animation_member"]
        ),
        "topology_match": (
            asset.rest_vertices.shape == asset.animated_bind_vertices.shape
            and asset.triangles.shape[1] == 3
        ),
        "vertex_count": asset.vertex_count,
        "triangle_count": int(asset.triangles.shape[0]),
        "bone_count": asset.bone_count,
        "weighted_bone_count": int(
            (asset.weights.sum(dim=0) > 0).sum().item()
        ),
        "unweighted_vertex_count": int(
            (asset.weights.sum(dim=1) == 0).sum().item()
        ),
        "weight_sum_max_error": float(weight_sum_error.max().item()),
        "uv_layer_count": 1 if asset.corner_uv.numel() else 0,
        "height": float(
            (asset.rest_vertices[:, 1].max() - asset.rest_vertices[:, 1].min())
            .item()
        ),
        "ground": float(asset.rest_vertices[:, 1].min().item()),
        "rest_pose_rms": float(
            torch.sqrt(rest_error.square().mean()).item()
        ),
        "rest_pose_max": float(rest_error.max().item()),
        "animation_frames": [
            int(value) for value in asset.animation_frames.tolist()
        ],
        "fps": int(asset.metadata["fps"]),
        "loop_skin_matrix_rms": float(
            torch.sqrt(loop_matrix_error.square().mean()).item()
        ),
        "loop_skin_matrix_max": float(loop_matrix_error.abs().max().item()),
    }

    influence_results = [
        evaluate_skinning(asset, count) for count in (None, 3, 4, 6)
    ]
    full_skinning = influence_results[0]
    selected_count = select_influence_count(influence_results[1:])
    if selected_count is None:
        selected_count = asset.bone_count
    selected_skinning = next(
        result for result in influence_results
        if result["influence_count"] == selected_count
    )
    selected_weights, _ = normalized_topk_weights(
        asset.weights, selected_count
    )
    surface_positions = _skin_in_frames(
        asset.animated_bind_vertices,
        selected_weights,
        asset.skin_matrices,
    )
    surface = {
        "cell_count": asset.vertex_count,
        "influence_count": selected_count,
        "finite": bool(torch.isfinite(surface_positions).all().item()),
        "loop_position_rms": float(torch.sqrt(
            (surface_positions[0] - surface_positions[-1])
            .square().sum(dim=1).mean()
        ).item()),
    }

    volume = build_h4_volume(
        asset,
        pitch=config.pitch,
        influence_count=selected_count,
        splat_radius_scale=config.volume_splat_radius_scale,
    )
    save_h4_volume(volume, output_directory / "volume.npz")
    volume_positions = _skin_in_frames(
        volume.points,
        volume.weights,
        asset.skin_matrices,
    )
    volume_metrics = {
        **volume.metadata,
        "valid_dominant_bone_fraction": float(
            (
                (volume.dominant_bone >= 0)
                & (volume.dominant_bone < asset.bone_count)
            ).double().mean().item()
        ),
        "finite": bool(torch.isfinite(volume_positions).all().item()),
        "loop_position_rms": float(torch.sqrt(
            (volume_positions[0] - volume_positions[-1])
            .square().sum(dim=1).mean()
        ).item()),
    }
    gates = acceptance_h4(
        provenance, full_skinning, selected_skinning, surface, volume_metrics
    )

    texture = load_base_color(config.archive_path)
    surface_uv = averaged_vertex_uv(asset).cpu().numpy()
    surface_colors = sample_texture(texture, surface_uv)
    volume_colors = sample_texture(texture, volume.uv.cpu().numpy())
    surface_frames = []
    volume_frames = []
    for frame_index, frame_number in enumerate(asset.animation_frames.tolist()):
        bones = asset.bone_endpoints[frame_index].cpu().numpy()
        surface_frames.append(render_colored_splats(
            surface_positions[frame_index].cpu().numpy(),
            surface_colors,
            bones,
            splat_radius=config.surface_splat_radius,
            size=config.image_size,
            label=f"surface frame {int(frame_number):02d} top-{selected_count}",
        ))
        volume_frames.append(render_colored_splats(
            volume_positions[frame_index].cpu().numpy(),
            volume_colors,
            bones,
            splat_radius=config.volume_splat_radius_scale * config.pitch,
            splat_scale=volume.splat_scale.cpu().numpy(),
            size=config.image_size,
            opacity=0.48,
            label=f"volume frame {int(frame_number):02d} cells {volume.cell_count}",
        ))
    save_gif(
        surface_frames,
        output_directory / "surface_animation.gif",
        duration_ms=config.gif_duration_ms,
    )
    save_gif(
        volume_frames,
        output_directory / "volume_animation.gif",
        duration_ms=config.gif_duration_ms,
    )
    milestones = [0, 7, 14, 22, 29]
    save_contact_sheet(
        [surface_frames[index] for index in milestones],
        output_directory / "surface_contact_sheet.png",
        columns=5,
    )
    save_contact_sheet(
        [volume_frames[index] for index in milestones],
        output_directory / "volume_contact_sheet.png",
        columns=5,
    )

    error = (surface_positions - asset.animation_vertices).norm(dim=-1)
    worst_frame = int(torch.sqrt(error.square().mean(dim=1)).argmax().item())
    diagnostic_colors = error_colors(
        error[worst_frame].cpu().numpy(), selected_skinning["animation_p99"]
    )
    render_colored_splats(
        asset.animation_vertices[worst_frame].cpu().numpy(),
        diagnostic_colors,
        asset.bone_endpoints[worst_frame].cpu().numpy(),
        splat_radius=config.surface_splat_radius,
        size=config.image_size,
        label=(
            f"top-{selected_count} error frame "
            f"{int(asset.animation_frames[worst_frame])}"
        ),
    ).save(output_directory / "surface_error.png")

    report = {
        "experiment": "H4",
        "description": "production rig import and variable-thickness target",
        "config": asdict(config),
        "provenance": provenance,
        "influence_results": influence_results,
        "selected_influence_count": selected_count,
        "surface": surface,
        "volume": volume_metrics,
        "acceptance": gates,
        "elapsed_seconds": time.time() - started,
    }
    (output_directory / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    verdict = "PASS" if gates["pass"] else "FAIL"
    (output_directory / "RUN.md").write_text(f"""# H4 run: {verdict}

- Source SHA-256: `{provenance['source_sha256']}`
- Vertices / triangles / bones: `{asset.vertex_count}` / `{asset.triangles.shape[0]}` / `{asset.bone_count}`
- Full skin RMS / max: `{full_skinning['animation_rms']:.9f}` / `{full_skinning['animation_max']:.9f}` m
- Selected influences: `{selected_count}`
- Selected RMS / p99 / max: `{selected_skinning['animation_rms']:.7f}` / `{selected_skinning['animation_p99']:.7f}` / `{selected_skinning['animation_max']:.7f}` m
- Volume cells / pitch: `{volume.cell_count}` / `{config.pitch}` m
- Extra-skeletal fraction / p95 distance: `{volume.metadata['extra_skeletal_fraction']:.4f}` / `{volume.metadata['bone_distance_p95']:.4f}` m
- Maximum volume splat radius: `{volume.metadata['maximum_world_splat_radius']:.5f}` m
- Acceptance: `{json.dumps(gates, sort_keys=True)}`
""")
    return report
