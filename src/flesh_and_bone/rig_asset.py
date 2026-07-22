"""Portable H4 rig asset loading, top-k weights, and linear skinning."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class RigAsset:
    """Canonical static mesh, skeleton, and evaluated animation oracle."""

    metadata: dict
    rest_vertices: torch.Tensor
    animated_bind_vertices: torch.Tensor
    triangles: torch.Tensor
    corner_uv: torch.Tensor
    weights: torch.Tensor
    bone_names: tuple[str, ...]
    bone_parents: torch.Tensor
    bind_matrices: torch.Tensor
    rest_bone_endpoints: torch.Tensor
    animation_frames: torch.Tensor
    animation_vertices: torch.Tensor
    skin_matrices: torch.Tensor
    bone_endpoints: torch.Tensor

    @property
    def vertex_count(self):
        return self.rest_vertices.shape[0]

    @property
    def bone_count(self):
        return len(self.bone_names)

    @property
    def frame_count(self):
        return self.animation_frames.shape[0]


def load_rig_asset(path, device=None, dtype=torch.float64):
    """Load the versioned Blender extraction without pickle/object arrays."""
    with np.load(Path(path), allow_pickle=False) as bundle:
        metadata = json.loads(str(bundle["metadata_json"].item()))
        if metadata.get("format") != "flesh-and-bone-h4-rig-v1":
            raise ValueError(f"unsupported rig format {metadata.get('format')}")

        def floating(name):
            return torch.as_tensor(
                np.array(bundle[name]), device=device, dtype=dtype
            )

        return RigAsset(
            metadata=metadata,
            rest_vertices=floating("rest_vertices"),
            animated_bind_vertices=floating("animated_bind_vertices"),
            triangles=torch.as_tensor(
                np.array(bundle["triangles"]), device=device, dtype=torch.long
            ),
            corner_uv=floating("corner_uv"),
            weights=floating("weights"),
            bone_names=tuple(str(value) for value in bundle["bone_names"]),
            bone_parents=torch.as_tensor(
                np.array(bundle["bone_parents"]),
                device=device,
                dtype=torch.long,
            ),
            bind_matrices=floating("bind_matrices"),
            rest_bone_endpoints=floating("rest_bone_endpoints"),
            animation_frames=torch.as_tensor(
                np.array(bundle["animation_frames"]),
                device=device,
                dtype=torch.long,
            ),
            animation_vertices=floating("animation_vertices"),
            skin_matrices=floating("skin_matrices"),
            bone_endpoints=floating("bone_endpoints"),
        )


def normalized_topk_weights(weights, influence_count=None):
    """Return normalized full/top-k weights and pre-renormalization retention."""
    if influence_count is None or influence_count >= weights.shape[1]:
        retained = weights.sum(dim=1)
        return weights / retained[:, None].clamp(min=1e-12), retained
    if influence_count < 1:
        raise ValueError("influence_count must be positive")
    values, indices = weights.topk(int(influence_count), dim=1)
    truncated = torch.zeros_like(weights)
    truncated.scatter_(1, indices, values)
    retained = values.sum(dim=1)
    return truncated / retained[:, None].clamp(min=1e-12), retained


def linear_skin(points, weights, skin_matrices):
    """Apply full frame/bone homogeneous skin matrices to canonical points."""
    if points.shape[0] != weights.shape[0]:
        raise ValueError("point and weight counts differ")
    if weights.shape[1] != skin_matrices.shape[1]:
        raise ValueError("weight and skin-matrix bone counts differ")
    homogeneous = torch.cat([
        points,
        torch.ones(
            points.shape[0], 1, device=points.device, dtype=points.dtype
        ),
    ], dim=1)
    transformed = torch.einsum(
        "fbij,vj->fbvi", skin_matrices, homogeneous
    )
    return torch.einsum("vb,fbvi->fvi", weights, transformed)[..., :3]


def evaluate_skinning(asset, influence_count=None):
    """Measure one influence arm against Blender-evaluated animation vertices."""
    weights, retained = normalized_topk_weights(
        asset.weights, influence_count
    )
    predicted = linear_skin(
        asset.animated_bind_vertices, weights, asset.skin_matrices
    )
    error = (predicted - asset.animation_vertices).norm(dim=-1)
    per_frame_rms = torch.sqrt(error.square().mean(dim=1))
    return {
        "influence_count": (
            asset.bone_count if influence_count is None else int(influence_count)
        ),
        "retained_weight_mean": float(retained.mean().item()),
        "retained_weight_min": float(retained.min().item()),
        "retained_weight_p01": float(torch.quantile(retained, 0.01).item()),
        "retained_weight_p05": float(torch.quantile(retained, 0.05).item()),
        "animation_rms": float(torch.sqrt(error.square().mean()).item()),
        "animation_p99": float(torch.quantile(error, 0.99).item()),
        "animation_max": float(error.max().item()),
        "per_frame_rms": [float(value) for value in per_frame_rms.tolist()],
        "finite": bool(torch.isfinite(predicted).all().item()),
    }


def averaged_vertex_uv(asset):
    """Average per-corner UV observations for inspection-only vertex colors."""
    uv_sum = torch.zeros(
        asset.vertex_count, 2,
        device=asset.corner_uv.device,
        dtype=asset.corner_uv.dtype,
    )
    count = torch.zeros(
        asset.vertex_count, 1,
        device=asset.corner_uv.device,
        dtype=asset.corner_uv.dtype,
    )
    indices = asset.triangles.reshape(-1)
    uv_sum.index_add_(0, indices, asset.corner_uv.reshape(-1, 2))
    count.index_add_(
        0,
        indices,
        torch.ones(
            indices.shape[0], 1,
            device=count.device,
            dtype=count.dtype,
        ),
    )
    return uv_sum / count.clamp(min=1)
