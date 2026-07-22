"""Contracts for portable rig loading and linear skinning primitives."""

import torch

from flesh_and_bone.rig_asset import linear_skin, normalized_topk_weights


def test_normalized_topk_weights_reports_retained_mass():
    weights = torch.tensor([
        [0.5, 0.3, 0.2],
        [0.7, 0.2, 0.1],
    ], dtype=torch.float64)
    top, retained = normalized_topk_weights(weights, 2)
    assert torch.allclose(
        retained, torch.tensor([0.8, 0.9], dtype=torch.float64)
    )
    assert torch.allclose(top.sum(dim=1), torch.ones(2, dtype=torch.float64))
    assert (top > 0).sum(dim=1).tolist() == [2, 2]


def test_linear_skin_blends_homogeneous_bone_transforms():
    points = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    weights = torch.tensor([[0.25, 0.75]], dtype=torch.float64)
    matrices = torch.eye(4, dtype=torch.float64).repeat(1, 2, 1, 1)
    matrices[0, 0, 0, 3] = 2.0
    matrices[0, 1, 1, 3] = 4.0
    skinned = linear_skin(points, weights, matrices)
    expected = torch.tensor([[[1.5, 3.0, 0.0]]], dtype=torch.float64)
    assert torch.allclose(skinned, expected)
