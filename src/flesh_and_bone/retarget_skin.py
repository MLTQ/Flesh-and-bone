"""Convert canonical rest-relative skeletal motion into H4 skin matrices."""

import torch


def _topological_order(parents):
    order = []
    visited = [False] * len(parents)
    for joint in range(len(parents)):
        stack = []
        node = joint
        while node >= 0 and not visited[node]:
            stack.append(node)
            node = int(parents[node])
        order.extend(reversed(stack))
        for node in stack:
            visited[node] = True
    return order


def canonical_motion_skin(asset, local_rotations, root_positions):
    """Return skin matrices and bone endpoints for canonical local rotations."""
    frames, bones = local_rotations.shape[:2]
    if local_rotations.shape != (frames, asset.bone_count, 3, 3):
        raise ValueError("local rotations and asset bone counts differ")
    if root_positions.shape != (frames, 3):
        raise ValueError("root positions must have shape [frames, 3]")
    parents = asset.bone_parents
    rest_heads = asset.rest_bone_endpoints[:, 0]
    rest_vectors = (
        asset.rest_bone_endpoints[:, 1]
        - asset.rest_bone_endpoints[:, 0]
    )
    world_rotations = torch.empty_like(local_rotations)
    world_positions = torch.empty(
        frames,
        bones,
        3,
        device=local_rotations.device,
        dtype=local_rotations.dtype,
    )
    identity = torch.eye(
        3, device=local_rotations.device, dtype=local_rotations.dtype
    ).expand(frames, 3, 3)
    for joint in _topological_order(parents.tolist()):
        parent = int(parents[joint])
        parent_rotation = (
            world_rotations[:, parent] if parent >= 0 else identity
        )
        world_rotations[:, joint] = (
            parent_rotation @ local_rotations[:, joint]
        )
        if parent < 0:
            world_positions[:, joint] = root_positions
        else:
            offset = rest_heads[joint] - rest_heads[parent]
            world_positions[:, joint] = (
                world_positions[:, parent]
                + torch.einsum("fij,j->fi", parent_rotation, offset)
            )

    pose = torch.zeros(
        frames,
        bones,
        4,
        4,
        device=local_rotations.device,
        dtype=local_rotations.dtype,
    )
    pose[..., 3, 3] = 1
    pose[..., :3, :3] = (
        world_rotations @ asset.bind_matrices[None, :, :3, :3]
    )
    pose[..., :3, 3] = world_positions
    skin_matrices = pose @ torch.linalg.inv(asset.bind_matrices)[None]
    tails = world_positions + torch.einsum(
        "fbij,bj->fbi", world_rotations, rest_vectors
    )
    bone_endpoints = torch.stack([world_positions, tails], dim=2)
    return skin_matrices, bone_endpoints


def palindrome_close(values, forward_frames):
    """Close a forward segment by returning through its interior in reverse."""
    forward_frames = int(forward_frames)
    if forward_frames < 3 or forward_frames > values.shape[0]:
        raise ValueError("forward_frames must select at least three input frames")
    reverse = torch.arange(
        forward_frames - 2,
        0,
        -1,
        device=values.device,
        dtype=torch.long,
    )
    return torch.cat([values[:forward_frames], values[reverse]], dim=0)
