"""Bounded dense-splat rendering for retargeted Kimodo review clips."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .h4_render import load_base_color, render_colored_splats, sample_texture
from .kimodo_client import KimodoReviewConfig
from .kimodo_diagnostics import bone_group_colors, hierarchy_segments
from .rig_asset import linear_skin


def _sample_indices(count: int, maximum: int) -> np.ndarray:
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    return np.unique(np.linspace(0, count - 1, maximum).round().astype(np.int64))


def _skin_selected(points, weights, skin, chunk_size: int = 4) -> np.ndarray:
    chunks = []
    for start in range(0, skin.shape[0], chunk_size):
        chunks.append(
            linear_skin(points, weights, skin[start:start + chunk_size])
            .detach()
            .cpu()
            .numpy()
        )
    return np.concatenate(chunks, axis=0)


def _contact_sheet(frames: list[Image.Image], path: Path):
    chosen = [frames[index] for index in _sample_indices(len(frames), 12)]
    thumb_size = 192
    columns = 4
    rows = int(np.ceil(len(chosen) / columns))
    sheet = Image.new("RGB", (columns * thumb_size, rows * thumb_size), (6, 8, 11))
    for index, frame in enumerate(chosen):
        thumb = frame.resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
        sheet.paste(
            thumb,
            ((index % columns) * thumb_size, (index // columns) * thumb_size),
        )
    sheet.save(path)


def render_review_preview(
    job_dir: Path,
    config: KimodoReviewConfig,
    asset,
    volume,
    skin: torch.Tensor,
    endpoints: np.ndarray,
    root: np.ndarray,
    fps: int,
    worst_frame: int,
) -> dict:
    """Write the texture GIF, contact sheet, and bone-group diagnostic."""
    cell_indices = _sample_indices(volume.cell_count, config.preview_cells)
    frame_indices = _sample_indices(skin.shape[0], config.preview_frames)
    selected_points = volume.points[cell_indices]
    selected_weights = volume.weights[cell_indices]
    selected_skin = skin[torch.as_tensor(frame_indices, dtype=torch.long)]
    skinned = _skin_selected(selected_points, selected_weights, selected_skin)
    texture = load_base_color(config.texture_archive)
    colors = sample_texture(
        texture, volume.uv[cell_indices].detach().cpu().numpy()
    )
    scales = volume.splat_scale[cell_indices].detach().cpu().numpy()
    all_segments = hierarchy_segments(
        endpoints, asset.bone_parents.detach().cpu().numpy(), include_count=22
    )
    splat_radius = (
        config.render_splat_radius_scale * float(volume.metadata["pitch"])
    )

    rendered = []
    for preview_index, source_frame in enumerate(frame_indices):
        follow = np.array(
            [root[source_frame, 0], 0.0, root[source_frame, 2]],
            dtype=np.float32,
        )
        rendered.append(
            render_colored_splats(
                skinned[preview_index] - follow,
                colors,
                bone_endpoints=all_segments[source_frame] - follow,
                splat_radius=splat_radius,
                splat_scale=scales,
                size=config.preview_size,
                label=f"frame {source_frame + 1}/{skin.shape[0]} · LBS review",
                opacity=0.88,
            )
        )
    duration_ms = max(20, round(1000 * skin.shape[0] / (fps * len(rendered))))
    rendered[0].save(
        job_dir / "character.gif",
        save_all=True,
        append_images=rendered[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )
    _contact_sheet(rendered, job_dir / "contact_sheet.png")

    worst_points = _skin_selected(
        selected_points,
        selected_weights,
        skin[worst_frame:worst_frame + 1],
        chunk_size=1,
    )[0]
    group_colors = bone_group_colors(
        volume.dominant_bone[cell_indices].detach().cpu().numpy(),
        asset.bone_names,
    )
    follow = np.array(
        [root[worst_frame, 0], 0.0, root[worst_frame, 2]], dtype=np.float32
    )
    diagnostic = render_colored_splats(
        worst_points - follow,
        group_colors,
        bone_endpoints=all_segments[worst_frame] - follow,
        splat_radius=splat_radius,
        splat_scale=scales,
        size=config.preview_size,
        label=f"worst pelvis frame {worst_frame + 1} · bone groups",
        opacity=0.90,
    )
    diagnostic.save(job_dir / "anatomy_frame.png")
    return {
        "preview_frame_count": int(len(rendered)),
        "source_frame_count": int(skin.shape[0]),
        "sampled_cell_count": int(len(cell_indices)),
        "source_cell_count": int(volume.cell_count),
        "size_px": int(config.preview_size),
        "splat_radius_m": float(splat_radius),
        "camera": "root-following XZ; world Y retained",
        "gif_frame_duration_ms": int(duration_ms),
    }
