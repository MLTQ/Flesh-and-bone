"""CPU-fast geometry contracts for H4 volume utilities."""

import numpy as np

from flesh_and_bone.h4_volume import occupancy_components, point_segment_distance


def test_point_segment_distance_clamps_to_bone_endpoints():
    bones = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    points = np.array([
        [0.5, 2.0, 0.0],
        [2.0, 0.0, 0.0],
    ])
    assert np.allclose(point_segment_distance(points, bones), [2.0, 1.0])


def test_occupancy_components_detects_separate_mass_and_enclosed_pocket():
    matrix = np.zeros((7, 7, 7), dtype=bool)
    matrix[1:6, 1:6, 1:6] = True
    matrix[3, 3, 3] = False
    matrix[0, 0, 0] = True
    result = occupancy_components(matrix)
    assert result["occupied_component_count"] == 2
    assert result["largest_occupied_component"] == 124
    assert result["largest_enclosed_empty_pocket"] == 1
