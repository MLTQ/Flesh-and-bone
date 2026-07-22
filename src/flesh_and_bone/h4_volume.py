"""Watertight H4 voxel volume and nearest-surface skin/material transfer."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
import torch
import trimesh

from .rig_asset import averaged_vertex_uv, normalized_topk_weights


@dataclass(frozen=True)
class H4Volume:
    """Fine filled humanoid cells carrying transferred rig/material state."""

    metadata: dict
    points: torch.Tensor
    weights: torch.Tensor
    dominant_bone: torch.Tensor
    nearest_surface_vertex: torch.Tensor
    uv: torch.Tensor
    bone_distance: torch.Tensor
    splat_scale: torch.Tensor

    @property
    def cell_count(self):
        return self.points.shape[0]


def point_segment_distance(points, endpoints):
    """Return each point's minimum Euclidean distance to any bone segment."""
    starts = endpoints[:, 0]
    vectors = endpoints[:, 1] - starts
    length_squared = np.maximum((vectors * vectors).sum(axis=1), 1e-12)
    relative = points[:, None] - starts[None]
    fraction = (
        (relative * vectors[None]).sum(axis=2) / length_squared[None]
    )
    fraction = np.clip(fraction, 0, 1)
    nearest = starts[None] + fraction[..., None] * vectors[None]
    return np.linalg.norm(points[:, None] - nearest, axis=2).min(axis=1)


def occupancy_components(matrix):
    """Measure occupied 6-connectivity and enclosed empty pocket sizes."""
    structure = ndimage.generate_binary_structure(3, 1)
    labels, count = ndimage.label(matrix, structure=structure)
    sizes = np.bincount(labels.reshape(-1))[1:]
    empty_labels, _ = ndimage.label(~matrix, structure=structure)
    boundary = np.concatenate([
        empty_labels[0].reshape(-1),
        empty_labels[-1].reshape(-1),
        empty_labels[:, 0].reshape(-1),
        empty_labels[:, -1].reshape(-1),
        empty_labels[:, :, 0].reshape(-1),
        empty_labels[:, :, -1].reshape(-1),
    ])
    external = set(int(value) for value in np.unique(boundary) if value)
    empty_sizes = np.bincount(empty_labels.reshape(-1))
    enclosed = [
        int(size) for label, size in enumerate(empty_sizes)
        if label and label not in external
    ]
    return {
        "occupied_component_count": int(count),
        "largest_occupied_component": int(sizes.max()) if sizes.size else 0,
        "largest_enclosed_empty_pocket": max(enclosed, default=0),
    }


def build_h4_volume(asset, pitch=0.025, influence_count=6,
                    splat_radius_scale=0.30):
    """Voxel-fill the imperfect source surface and transfer nearby state."""
    vertices = asset.rest_vertices.detach().cpu().numpy()
    faces = asset.triangles.detach().cpu().numpy()
    surface = trimesh.Trimesh(
        vertices=vertices, faces=faces, process=False
    )
    voxel = surface.voxelized(pitch=float(pitch), method="subdivide").fill()
    points = np.asarray(voxel.points, dtype=np.float64)
    matrix = np.asarray(voxel.matrix, dtype=bool)
    components = occupancy_components(matrix)

    tree = cKDTree(vertices)
    nearest_distance, nearest = tree.query(points, k=1, workers=-1)
    source_weights, retained = normalized_topk_weights(
        asset.weights, influence_count
    )
    weights = source_weights.detach().cpu().numpy()[nearest]
    vertex_uv = averaged_vertex_uv(asset).detach().cpu().numpy()
    uv = vertex_uv[nearest]
    dominant = weights.argmax(axis=1)
    bone_distance = point_segment_distance(
        points, asset.rest_bone_endpoints.detach().cpu().numpy()
    )
    thickness_scale = np.clip(bone_distance / 0.18, 0, 1)
    splat_scale = 0.75 + 0.40 * thickness_scale
    maximum_world_radius = (
        float(splat_radius_scale) * float(pitch) * float(splat_scale.max())
    )
    metadata = {
        "format": "flesh-and-bone-h4-volume-v1",
        "source_sha256": asset.metadata["source_sha256"],
        "pitch": float(pitch),
        "influence_count": int(influence_count),
        "splat_radius_scale": float(splat_radius_scale),
        "cell_count": int(points.shape[0]),
        "grid_shape": [int(value) for value in matrix.shape],
        "surface_watertight": bool(surface.is_watertight),
        "surface_winding_consistent": bool(surface.is_winding_consistent),
        "nearest_surface_distance_mean": float(nearest_distance.mean()),
        "nearest_surface_distance_max": float(nearest_distance.max()),
        "weight_sum_max_error": float(np.abs(weights.sum(axis=1) - 1).max()),
        "source_retained_weight_mean": float(retained.mean().item()),
        "occupied_component_count": components["occupied_component_count"],
        "largest_occupied_component": components["largest_occupied_component"],
        "largest_enclosed_empty_pocket": components[
            "largest_enclosed_empty_pocket"
        ],
        "enclosed_pocket_fraction": (
            components["largest_enclosed_empty_pocket"] / points.shape[0]
        ),
        "extra_skeletal_fraction": float((bone_distance > 0.08).mean()),
        "bone_distance_p95": float(np.quantile(bone_distance, 0.95)),
        "bone_distance_max": float(bone_distance.max()),
        "splat_scale_min": float(splat_scale.min()),
        "splat_scale_max": float(splat_scale.max()),
        "maximum_world_splat_radius": maximum_world_radius,
    }
    volume = H4Volume(
        metadata=metadata,
        points=torch.as_tensor(points, dtype=asset.rest_vertices.dtype),
        weights=torch.as_tensor(weights, dtype=asset.weights.dtype),
        dominant_bone=torch.as_tensor(dominant, dtype=torch.long),
        nearest_surface_vertex=torch.as_tensor(nearest, dtype=torch.long),
        uv=torch.as_tensor(uv, dtype=asset.corner_uv.dtype),
        bone_distance=torch.as_tensor(
            bone_distance, dtype=asset.rest_vertices.dtype
        ),
        splat_scale=torch.as_tensor(
            splat_scale, dtype=asset.rest_vertices.dtype
        ),
    )
    return volume


def save_h4_volume(volume, path):
    """Write a versioned, non-pickle compressed volume artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        metadata_json=np.asarray(json.dumps(volume.metadata)),
        points=volume.points.detach().cpu().numpy(),
        weights=volume.weights.detach().cpu().numpy(),
        dominant_bone=volume.dominant_bone.detach().cpu().numpy(),
        nearest_surface_vertex=(
            volume.nearest_surface_vertex.detach().cpu().numpy()
        ),
        uv=volume.uv.detach().cpu().numpy(),
        bone_distance=volume.bone_distance.detach().cpu().numpy(),
        splat_scale=volume.splat_scale.detach().cpu().numpy(),
    )


def load_h4_volume(path, device=None, dtype=torch.float32):
    """Load the versioned derived volume into Torch tensors."""
    with np.load(Path(path), allow_pickle=False) as bundle:
        metadata = json.loads(str(bundle["metadata_json"].item()))
        if metadata.get("format") != "flesh-and-bone-h4-volume-v1":
            raise ValueError("unsupported H4 volume format")
        floating = lambda name: torch.as_tensor(
            np.array(bundle[name]), device=device, dtype=dtype
        )
        return H4Volume(
            metadata=metadata,
            points=floating("points"),
            weights=floating("weights"),
            dominant_bone=torch.as_tensor(
                np.array(bundle["dominant_bone"]),
                device=device,
                dtype=torch.long,
            ),
            nearest_surface_vertex=torch.as_tensor(
                np.array(bundle["nearest_surface_vertex"]),
                device=device,
                dtype=torch.long,
            ),
            uv=floating("uv"),
            bone_distance=floating("bone_distance"),
            splat_scale=floating("splat_scale"),
        )
