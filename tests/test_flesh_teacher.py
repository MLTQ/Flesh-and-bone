"""CPU-fast contracts for H5 sparse neighbor mechanics."""

import torch

from flesh_and_bone.flesh_teacher import (
    build_voxel_graph,
    neighbor_mean_difference,
    periodic_acceleration,
)


def test_voxel_graph_and_neighbor_mean_on_three_cell_line():
    points = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ])
    graph = build_voxel_graph(points, pitch=1.0)
    values = points.clone()
    difference = neighbor_mean_difference(values, graph)
    assert graph.component_count == 1
    assert graph.degree.tolist() == [1.0, 2.0, 1.0]
    assert torch.allclose(
        difference[:, 0], torch.tensor([1.0, 0.0, -1.0])
    )


def test_periodic_acceleration_is_zero_for_static_cycle():
    positions = torch.ones(5, 2, 3)
    acceleration = periodic_acceleration(positions, fps=30)
    assert torch.equal(acceleration, torch.zeros_like(positions))
