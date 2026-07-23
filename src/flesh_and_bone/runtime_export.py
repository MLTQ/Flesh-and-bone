"""Export compact, versioned assets for the native Flesh-and-Bone runner."""

from __future__ import annotations

import json
from pathlib import Path
import struct

import numpy as np
import torch

from .h4_render import load_base_color, sample_texture
from .h4_volume import load_h4_volume
from .rig_asset import load_rig_asset


BODY_MAGIC = b"FNB1"
MODEL_MAGIC = b"FNM1"
BODY_HEADER = struct.Struct("<4sIIIIfff")
MODEL_HEADER = struct.Struct("<4sIIIfff")
BODY_VERSION = 1
MODEL_VERSION = 1


def _six_neighbors(points: np.ndarray, pitch: float) -> np.ndarray:
    """Return fixed-width six-neighbor indices with -1 padding."""
    integer = np.rint(
        (points - points.min(axis=0)) / float(pitch)
    ).astype(np.int64)
    lookup = {tuple(value): index for index, value in enumerate(integer)}
    neighbors = np.full((points.shape[0], 8), -1, dtype="<i4")
    for index, coordinate in enumerate(integer):
        selected = []
        for axis in range(3):
            for direction in (-1, 1):
                neighbor = coordinate.copy()
                neighbor[axis] += direction
                target = lookup.get(tuple(neighbor))
                if target is not None:
                    selected.append(target)
        neighbors[index, :len(selected)] = selected
    return neighbors


def _top_six(weights: np.ndarray):
    """Pack six nonzero influences into eight aligned lanes."""
    order = np.argsort(-weights, axis=1, kind="stable")[:, :6]
    values = np.take_along_axis(weights, order, axis=1)
    values /= np.maximum(values.sum(axis=1, keepdims=True), 1e-12)
    indices = np.zeros((weights.shape[0], 8), dtype="<u2")
    packed = np.zeros((weights.shape[0], 8), dtype="<f4")
    indices[:, :6] = order.astype("<u2")
    packed[:, :6] = values.astype("<f4")
    return indices, packed


def export_runtime_body(
    rig_path,
    volume_path,
    texture_archive,
    output_path,
    radius_scale=0.50,
    render_order_seed=804,
):
    """Export one physical-resolution body, graph, material, and walk rig."""
    asset = load_rig_asset(rig_path, dtype=torch.float32)
    volume = load_h4_volume(volume_path, dtype=torch.float32)
    points = volume.points.numpy().astype("<f4", copy=False)
    pitch = float(volume.metadata["pitch"])
    cell_count = int(points.shape[0])
    bone_count = int(asset.bone_count)
    skin_matrices = asset.skin_matrices[:-1].numpy()
    frame_count = int(skin_matrices.shape[0])
    point4 = np.concatenate([
        points,
        np.ones((cell_count, 1), dtype="<f4"),
    ], axis=1)
    skin_indices, skin_weights = _top_six(volume.weights.numpy())
    softness = np.clip(volume.bone_distance.numpy() / 0.18, 0, 1)
    stiffness = 1200.0 * (1 - softness) + 300.0 * softness
    damping = 2 * 0.22 * np.sqrt(stiffness)
    material = np.stack([
        softness,
        stiffness,
        damping,
        volume.splat_scale.numpy(),
    ], axis=1).astype("<f4")
    neighbors = _six_neighbors(points, pitch)
    texture = load_base_color(texture_archive)
    rgb = np.clip(
        np.rint(sample_texture(texture, volume.uv.numpy()) * 255),
        0,
        255,
    ).astype(np.uint8)
    colors = np.concatenate([
        rgb,
        np.full((cell_count, 1), 255, dtype=np.uint8),
    ], axis=1)
    generator = np.random.default_rng(int(render_order_seed))
    render_order = generator.permutation(cell_count).astype("<u4")
    metal_matrices = np.ascontiguousarray(
        skin_matrices.transpose(0, 1, 3, 2),
        dtype="<f4",
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        output.write(BODY_HEADER.pack(
            BODY_MAGIC,
            BODY_VERSION,
            cell_count,
            bone_count,
            frame_count,
            pitch,
            float(radius_scale) * pitch,
            0.0,
        ))
        for value in (
            point4,
            skin_indices,
            skin_weights,
            material,
            neighbors,
            colors,
            render_order,
            metal_matrices,
        ):
            output.write(value.tobytes(order="C"))
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "cell_count": cell_count,
        "bone_count": bone_count,
        "frame_count": frame_count,
        "pitch": pitch,
        "base_radius": float(radius_scale) * pitch,
        "neighbor_count_mean": float((neighbors >= 0).sum(axis=1).mean()),
        "neighbor_count_min": int((neighbors >= 0).sum(axis=1).min()),
        "neighbor_count_max": int((neighbors >= 0).sum(axis=1).max()),
    }


def export_runtime_model(checkpoint_path, metrics_path, output_path):
    """Export one H7C residual and its exact H6C backbone coefficients."""
    state = torch.load(
        Path(checkpoint_path), map_location="cpu", weights_only=True
    )
    metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    coefficients = np.asarray(
        metrics["backbone"]["coefficients"], dtype="<f4"
    )
    arrays = [
        coefficients,
        state["coefficient_maxima"].numpy().astype("<f4"),
        state["network.0.weight"].numpy().astype("<f4"),
        state["network.0.bias"].numpy().astype("<f4"),
        state["network.2.weight"].numpy().astype("<f4"),
        state["network.2.bias"].numpy().astype("<f4"),
        state["network.4.weight"].numpy().astype("<f4"),
        state["network.4.bias"].numpy().astype("<f4"),
    ]
    hidden = int(arrays[2].shape[0])
    if arrays[2].shape != (hidden, 5):
        raise ValueError("runtime model requires a five-input first layer")
    if arrays[4].shape != (hidden, hidden):
        raise ValueError("runtime model hidden layers must be square")
    if arrays[6].shape != (2, hidden):
        raise ValueError("runtime model requires two coefficient outputs")

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        output.write(MODEL_HEADER.pack(
            MODEL_MAGIC,
            MODEL_VERSION,
            hidden,
            5,
            12.0,
            0.0125,
            0.0,
        ))
        for value in arrays:
            output.write(value.tobytes(order="C"))
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "hidden_channels": hidden,
        "learned_parameters": int(sum(
            value.size for value in arrays[2:]
        )),
        "backbone_coefficients": coefficients.tolist(),
        "coefficient_maxima": arrays[1].tolist(),
        "reference_pitch": 0.0125,
        "acceleration_cap": 12.0,
    }


def export_runtime_bundle(
    output_directory,
    checkpoint_path,
    metrics_path,
    rig_path,
    volume_paths,
    texture_archive,
):
    """Export all resolution profiles and one shared model with a manifest."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    model = export_runtime_model(
        checkpoint_path,
        metrics_path,
        output / "h7c_seed7.fnm",
    )
    bodies = [
        export_runtime_body(
            rig_path,
            volume_path,
            texture_archive,
            output / f"body_{load_h4_volume(volume_path).cell_count}.fnb",
        )
        for volume_path in volume_paths
    ]
    manifest = {
        "format": "flesh-and-bone-runtime-bundle-v1",
        "model": model,
        "bodies": bodies,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
