"""Texture sampling and inspection rendering for fine H4 rigged splats."""

from io import BytesIO
import math
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image, ImageDraw


SKELETON_COLOR = (238, 177, 62)


def load_base_color(archive_path):
    """Load the standalone base-color texture directly from the source zip."""
    with zipfile.ZipFile(Path(archive_path)) as archive:
        candidates = [
            name for name in archive.namelist()
            if name.endswith("_texture_0.png")
        ]
        if len(candidates) != 1:
            raise ValueError(f"expected one base-color texture, found {candidates}")
        with archive.open(candidates[0]) as source:
            image = Image.open(BytesIO(source.read())).convert("RGB")
    return image


def sample_texture(image, uv):
    """Bilinearly sample repeated Blender UVs into floating RGB colors."""
    pixels = np.asarray(image, dtype=np.float32) / 255
    uv = np.asarray(uv, dtype=np.float64)
    u = np.mod(uv[:, 0], 1.0) * (image.width - 1)
    v = (1 - np.mod(uv[:, 1], 1.0)) * (image.height - 1)
    x0 = np.floor(u).astype(np.int64)
    y0 = np.floor(v).astype(np.int64)
    x1 = np.minimum(x0 + 1, image.width - 1)
    y1 = np.minimum(y0 + 1, image.height - 1)
    tx = (u - x0)[:, None]
    ty = (v - y0)[:, None]
    top = pixels[y0, x0] * (1 - tx) + pixels[y0, x1] * tx
    bottom = pixels[y1, x0] * (1 - tx) + pixels[y1, x1] * tx
    return top * (1 - ty) + bottom * ty


def _camera(points, size, world_height=1.95, yaw=0.30):
    cosine, sine = math.cos(yaw), math.sin(yaw)
    view_x = cosine * points[:, 0] + sine * points[:, 2]
    depth = -sine * points[:, 0] + cosine * points[:, 2]
    scale = size / world_height
    screen = np.stack([
        size * 0.5 + view_x * scale,
        size * 0.94 - points[:, 1] * scale,
    ], axis=1)
    return screen, depth, scale


def render_colored_splats(points, colors, bone_endpoints=None, splat_radius=0.007,
                          splat_scale=None, size=480, label=None,
                          world_height=1.95, opacity=0.82):
    """Render depth-sorted colored Gaussian cells and optional rig segments."""
    background = np.zeros((size, size, 3), dtype=np.float32)
    background[:] = np.array([0.025, 0.032, 0.040], dtype=np.float32)
    image = Image.fromarray((background * 255).astype(np.uint8), "RGB")
    if bone_endpoints is not None:
        draw = ImageDraw.Draw(image)
        endpoints = np.asarray(bone_endpoints).reshape(-1, 3)
        projected, _, _ = _camera(
            endpoints, size, world_height=world_height
        )
        projected = projected.reshape(-1, 2, 2)
        for segment in projected:
            draw.line(
                [tuple(segment[0]), tuple(segment[1])],
                fill=SKELETON_COLOR,
                width=2,
            )
    canvas = np.asarray(image, dtype=np.float32) / 255
    points = np.asarray(points, dtype=np.float32)
    colors = np.clip(np.asarray(colors, dtype=np.float32), 0, 1)
    scales = (
        np.ones(points.shape[0], dtype=np.float32)
        if splat_scale is None else np.asarray(splat_scale, dtype=np.float32)
    )
    screen, depth, camera_scale = _camera(
        points, size, world_height=world_height
    )
    sigma_values = np.maximum(0.75, splat_radius * camera_scale * scales)
    for index in np.argsort(depth):
        sigma = float(sigma_values[index])
        radius = int(math.ceil(2.7 * sigma))
        cx, cy = screen[index]
        if cx < -radius or cy < -radius or cx >= size + radius or cy >= size + radius:
            continue
        x0, x1 = max(0, int(cx) - radius), min(size, int(cx) + radius + 1)
        y0, y1 = max(0, int(cy) - radius), min(size, int(cy) + radius + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        gaussian = np.exp(
            -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)
        )
        alpha = (opacity * gaussian)[..., None]
        canvas[y0:y1, x0:x1] = (
            colors[index][None, None] * alpha
            + canvas[y0:y1, x0:x1] * (1 - alpha)
        )
    output = Image.fromarray(
        (np.clip(canvas, 0, 1) * 255).astype(np.uint8), "RGB"
    )
    if label:
        draw = ImageDraw.Draw(output)
        draw.rectangle((5, 5, 8 + 7 * len(label), 23), fill=(6, 8, 11))
        draw.text((9, 8), label, fill=(235, 238, 240))
    return output


def error_colors(error, ceiling):
    """Map zero-to-ceiling metric error from blue through white to red."""
    value = np.clip(np.asarray(error) / max(float(ceiling), 1e-12), 0, 1)
    return np.stack([
        value,
        0.25 + 0.65 * (1 - np.abs(2 * value - 1)),
        1 - value,
    ], axis=1)
