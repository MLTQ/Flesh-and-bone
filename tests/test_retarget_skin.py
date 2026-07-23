"""CPU-fast identity and closure contracts for canonical motion skinning."""

from types import SimpleNamespace

import torch

from flesh_and_bone.retarget_skin import (
    canonical_motion_skin,
    palindrome_close,
)


def test_identity_canonical_motion_produces_identity_skin():
    asset = SimpleNamespace(
        bone_count=2,
        bone_parents=torch.tensor([-1, 0]),
        rest_bone_endpoints=torch.tensor([
            [[0.0, 1.0, 0.0], [0.0, 1.5, 0.0]],
            [[0.0, 1.5, 0.0], [0.0, 2.0, 0.0]],
        ]),
        bind_matrices=torch.tensor([
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 1.0],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]],
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 1.5],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]],
        ]),
    )
    local = torch.eye(3).expand(1, 2, 3, 3).clone()
    skin, endpoints = canonical_motion_skin(
        asset, local, torch.tensor([[0.0, 1.0, 0.0]])
    )
    assert torch.allclose(skin, torch.eye(4).expand(1, 2, 4, 4))
    assert torch.allclose(endpoints, asset.rest_bone_endpoints[None])


def test_palindrome_closure_has_no_duplicate_turnaround_or_wrap_endpoint():
    values = torch.arange(5)
    closed = palindrome_close(values, 5)
    assert closed.tolist() == [0, 1, 2, 3, 4, 3, 2, 1]
