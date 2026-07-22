"""Orthographic Gaussian-splat evidence rendering for particle creatures."""

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


CHECKER_COLORS = np.array([
    [0.07, 0.19, 0.27],
    [0.88, 0.94, 0.84],
], dtype=np.float32)
UNCOMMITTED_COLOR = np.array([1.0, 0.34, 0.06], dtype=np.float32)
SKELETON_COLOR = (238, 177, 62)


def _camera(points, size, world_extent=2.65, tilt=0.28):
    cosine, sine = math.cos(tilt), math.sin(tilt)
    rotation = np.array([
        [cosine, 0.0, sine],
        [0.0, 1.0, 0.0],
        [-sine, 0.0, cosine],
    ], dtype=np.float32)
    view = points @ rotation.T
    scale = size / world_extent
    screen = np.stack([
        size * 0.5 + view[:, 0] * scale,
        size * 0.5 - view[:, 1] * scale,
    ], axis=1)
    return screen, view[:, 2], scale


def render_frame(particles, frame, size=320, splat_radius_world=0.0434,
                 label=None, world_extent=2.65, minimum_sigma_pixels=2.2):
    """Render active particles as depth-sorted material-scaled splats."""
    background = np.zeros((size, size, 3), dtype=np.float32)
    background[:] = np.array([0.025, 0.032, 0.040], dtype=np.float32)
    image = Image.fromarray((background * 255).astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(image)
    endpoints = frame.endpoints.detach().cpu().numpy().reshape(-1, 3)
    projected_endpoints, _, _ = _camera(
        endpoints, size, world_extent=world_extent
    )
    projected_endpoints = projected_endpoints.reshape(
        frame.endpoints.shape[0], 2, 2
    )
    for segment in projected_endpoints:
        draw.line([tuple(segment[0]), tuple(segment[1])], fill=SKELETON_COLOR, width=3)
    canvas = np.asarray(image, dtype=np.float32) / 255

    active = particles.active.detach().cpu().numpy().astype(bool)
    if active.any():
        points = particles.positions.detach().cpu().numpy()[active]
        committed = particles.committed.detach().cpu().numpy()[active].astype(bool)
        checker = particles.checker.detach().cpu().numpy()[active]
        splat_scale = particles.splat_scale.detach().cpu().numpy()[active]
        screen, depth, scale = _camera(
            points, size, world_extent=world_extent
        )
        sigma_values = np.maximum(
            float(minimum_sigma_pixels),
            float(splat_radius_world) * scale * splat_scale,
        )
        order = np.argsort(depth)
        for index in order:
            sigma = sigma_values[index]
            radius = int(math.ceil(3 * sigma))
            cx, cy = screen[index]
            if cx < -radius or cy < -radius or cx >= size + radius or cy >= size + radius:
                continue
            x0, x1 = max(0, int(cx) - radius), min(size, int(cx) + radius + 1)
            y0, y1 = max(0, int(cy) - radius), min(size, int(cy) + radius + 1)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            gaussian = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
            opacity = 0.86 if committed[index] else 0.58
            alpha = (opacity * gaussian)[..., None]
            color = (
                CHECKER_COLORS[checker[index]]
                if committed[index] else UNCOMMITTED_COLOR
            )
            canvas[y0:y1, x0:x1] = (
                color[None, None] * alpha
                + canvas[y0:y1, x0:x1] * (1 - alpha)
            )
    output = Image.fromarray((np.clip(canvas, 0, 1) * 255).astype(np.uint8), "RGB")
    if label:
        draw = ImageDraw.Draw(output)
        draw.rectangle((5, 5, 5 + 7 * len(label), 23), fill=(6, 8, 11))
        draw.text((9, 8), label, fill=(235, 238, 240))
    return output


def save_gif(frames, path, duration_ms=70):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        path, save_all=True, append_images=frames[1:], duration=duration_ms,
        loop=0, disposal=2,
    )


def save_contact_sheet(frames, path, columns=3):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = math.ceil(len(frames) / columns)
    width, height = frames[0].size
    sheet = Image.new("RGB", (columns * width, rows * height), (6, 8, 10))
    for index, frame in enumerate(frames):
        sheet.paste(frame, ((index % columns) * width, (index // columns) * height))
    sheet.save(path)
