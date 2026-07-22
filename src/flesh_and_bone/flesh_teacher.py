"""Sparse graph-elastic soft-tissue teacher driven by H4 LBS motion."""

from dataclasses import dataclass

import numpy as np
import torch

from .rig_asset import linear_skin


@dataclass(frozen=True)
class ElasticTeacherConfig:
    fps: float = 30.0
    substeps: int = 4
    warmup_cycles: int = 30
    near_stiffness: float = 1200.0
    far_stiffness: float = 300.0
    softening_distance: float = 0.18
    damping_ratio: float = 0.22
    neighbor_coupling: float = 300.0


@dataclass(frozen=True)
class VoxelGraph:
    source: torch.Tensor
    target: torch.Tensor
    degree: torch.Tensor
    component_count: int


@dataclass(frozen=True)
class TeacherTrajectory:
    residual: torch.Tensor
    velocity: torch.Tensor
    acceleration: torch.Tensor
    neighbor_residual: torch.Tensor
    neighbor_velocity: torch.Tensor
    lbs_positions: torch.Tensor
    lbs_acceleration: torch.Tensor
    stiffness: torch.Tensor
    final_residual: torch.Tensor
    final_velocity: torch.Tensor


def build_voxel_graph(points, pitch):
    """Build directed six-neighbor edges from a filled regular voxel lattice."""
    device = points.device
    coordinates = points.detach().cpu().numpy()
    integer = np.rint(
        (coordinates - coordinates.min(axis=0)) / float(pitch)
    ).astype(np.int64)
    lookup = {tuple(value): index for index, value in enumerate(integer)}
    source, target = [], []
    for index, coordinate in enumerate(integer):
        for axis in range(3):
            for direction in (-1, 1):
                neighbor = coordinate.copy()
                neighbor[axis] += direction
                selected = lookup.get(tuple(neighbor))
                if selected is not None:
                    source.append(index)
                    target.append(selected)
    source = torch.as_tensor(source, device=device, dtype=torch.long)
    target = torch.as_tensor(target, device=device, dtype=torch.long)
    degree = torch.bincount(source, minlength=points.shape[0]).to(points.dtype)

    seen = set()
    adjacency = [[] for _ in range(points.shape[0])]
    for left, right in zip(source.cpu().tolist(), target.cpu().tolist()):
        adjacency[left].append(right)
    components = 0
    for seed in range(points.shape[0]):
        if seed in seen:
            continue
        components += 1
        stack = [seed]
        seen.add(seed)
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
    return VoxelGraph(
        source=source,
        target=target,
        degree=degree,
        component_count=components,
    )


def neighbor_mean_difference(values, graph):
    """Average `neighbor - self` over directed six-neighbor edges."""
    result = torch.zeros_like(values)
    result.index_add_(
        0,
        graph.source,
        values[graph.target] - values[graph.source],
    )
    return result / graph.degree[:, None].clamp(min=1)


def volume_lbs_cycle(asset, volume):
    """Skin the 29 unique walk phases and compute periodic LBS acceleration."""
    unique_frames = asset.skin_matrices[:-1]
    positions = torch.cat([
        linear_skin(
            volume.points,
            volume.weights,
            unique_frames[index:index + 1],
        )
        for index in range(unique_frames.shape[0])
    ], dim=0)
    dt = 1.0 / float(asset.metadata["fps"])
    acceleration = (
        torch.roll(positions, -1, dims=0)
        - 2 * positions
        + torch.roll(positions, 1, dims=0)
    ) / (dt * dt)
    return positions, acceleration


def simulate_teacher(asset, volume, graph, config=None):
    """Warm to a limit cycle, then capture one cycle at every substep."""
    config = config or ElasticTeacherConfig()
    lbs_positions, lbs_acceleration = volume_lbs_cycle(asset, volume)
    distance = (volume.bone_distance / config.softening_distance).clamp(0, 1)
    stiffness = (
        config.near_stiffness * (1 - distance)
        + config.far_stiffness * distance
    )
    damping = 2 * config.damping_ratio * torch.sqrt(stiffness)
    residual = torch.zeros_like(volume.points)
    velocity = torch.zeros_like(residual)
    substep = 1.0 / (config.fps * config.substeps)

    def advance(phase, capture=None):
        nonlocal residual, velocity
        for local_step in range(config.substeps):
            neighbor_residual = neighbor_mean_difference(residual, graph)
            neighbor_velocity = neighbor_mean_difference(velocity, graph)
            acceleration = (
                -stiffness[:, None] * residual
                - damping[:, None] * velocity
                + config.neighbor_coupling * neighbor_residual
                - lbs_acceleration[phase]
            )
            if capture is not None:
                capture["residual"].append(residual.clone())
                capture["velocity"].append(velocity.clone())
                capture["acceleration"].append(acceleration.clone())
                capture["neighbor_residual"].append(
                    neighbor_residual.clone()
                )
                capture["neighbor_velocity"].append(
                    neighbor_velocity.clone()
                )
            velocity = velocity + substep * acceleration
            residual = residual + substep * velocity

    with torch.no_grad():
        for _ in range(config.warmup_cycles):
            for phase in range(lbs_positions.shape[0]):
                advance(phase)
        captured = {
            name: [] for name in (
                "residual", "velocity", "acceleration",
                "neighbor_residual", "neighbor_velocity",
            )
        }
        for phase in range(lbs_positions.shape[0]):
            advance(phase, captured)
    shape = (lbs_positions.shape[0], config.substeps, volume.cell_count, 3)
    return TeacherTrajectory(
        residual=torch.stack(captured["residual"]).reshape(shape),
        velocity=torch.stack(captured["velocity"]).reshape(shape),
        acceleration=torch.stack(captured["acceleration"]).reshape(shape),
        neighbor_residual=torch.stack(
            captured["neighbor_residual"]
        ).reshape(shape),
        neighbor_velocity=torch.stack(
            captured["neighbor_velocity"]
        ).reshape(shape),
        lbs_positions=lbs_positions,
        lbs_acceleration=lbs_acceleration,
        stiffness=stiffness,
        final_residual=residual,
        final_velocity=velocity,
    )
