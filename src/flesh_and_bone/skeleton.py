"""Five-bone articulated H scaffold and bone-coordinate geometry."""

from dataclasses import dataclass

import torch


BONE_NAMES = (
    "left_upper",
    "left_lower",
    "bridge",
    "right_upper",
    "right_lower",
)


@dataclass(frozen=True)
class ScaffoldFrame:
    """World-space endpoints for an ordered set of bone segments."""

    endpoints: torch.Tensor

    def __post_init__(self):
        if (
            self.endpoints.ndim != 3
            or self.endpoints.shape[0] < 1
            or self.endpoints.shape[1:] != (2, 3)
        ):
            raise ValueError("scaffold endpoints must be shaped (bones, 2, 3)")

    @property
    def bone_count(self):
        return self.endpoints.shape[0]


def _rotation_x(angle):
    zero, one = torch.zeros_like(angle), torch.ones_like(angle)
    cosine, sine = torch.cos(angle), torch.sin(angle)
    return torch.stack([
        one, zero, zero,
        zero, cosine, -sine,
        zero, sine, cosine,
    ]).reshape(3, 3)


def _rotation_z(angle):
    zero, one = torch.zeros_like(angle), torch.ones_like(angle)
    cosine, sine = torch.cos(angle), torch.sin(angle)
    return torch.stack([
        cosine, -sine, zero,
        sine, cosine, zero,
        zero, zero, one,
    ]).reshape(3, 3)


class HScaffold:
    """Connected double-Y skeleton with bounded procedural motion."""

    def __init__(self, half_width=0.55, arm_length=0.90):
        self.half_width = float(half_width)
        self.arm_length = float(arm_length)

    def frame(self, time=0.0, device=None, dtype=torch.float32):
        time = torch.as_tensor(time, device=device, dtype=dtype)
        half_width = torch.as_tensor(self.half_width, device=device, dtype=dtype)
        arm_length = torch.as_tensor(self.arm_length, device=device, dtype=dtype)
        left = torch.stack([-half_width, time.new_zeros(()), time.new_zeros(())])
        right = torch.stack([half_width, time.new_zeros(()), time.new_zeros(())])

        upper = torch.stack([time.new_zeros(()), arm_length, time.new_zeros(())])
        lower = torch.stack([time.new_zeros(()), -arm_length, time.new_zeros(())])
        left_angle = 0.16 * torch.sin(time)
        right_angle = 0.14 * torch.sin(time + 0.8) - 0.14 * torch.sin(time.new_tensor(0.8))
        depth_angle = 0.10 * torch.sin(0.73 * time)
        left_upper = left + _rotation_x(depth_angle) @ (_rotation_z(left_angle) @ upper)
        left_lower = left + _rotation_x(-depth_angle) @ (_rotation_z(-0.75 * left_angle) @ lower)
        right_upper = right + _rotation_x(-0.8 * depth_angle) @ (_rotation_z(-right_angle) @ upper)
        right_lower = right + _rotation_x(0.9 * depth_angle) @ (_rotation_z(0.70 * right_angle) @ lower)

        endpoints = torch.stack([
            torch.stack([left, left_upper]),
            torch.stack([left, left_lower]),
            torch.stack([left, right]),
            torch.stack([right, right_upper]),
            torch.stack([right, right_lower]),
        ])
        global_rotation = _rotation_z(0.22 * torch.sin(0.47 * time)) @ _rotation_x(
            0.10 * torch.sin(0.61 * time)
        )
        endpoints = endpoints @ global_rotation.T
        endpoints[..., 2] += 0.07 * torch.sin(0.83 * time)
        return ScaffoldFrame(endpoints)


def segment_projection(points, endpoints):
    """Project points onto every clamped bone segment."""
    starts = endpoints[:, 0]
    vectors = endpoints[:, 1] - starts
    lengths_squared = vectors.square().sum(dim=-1).clamp(min=1e-10)
    relative = points[:, None, :] - starts[None, :, :]
    fraction = (relative * vectors[None]).sum(dim=-1) / lengths_squared[None]
    fraction = fraction.clamp(0, 1)
    nearest = starts[None] + fraction[..., None] * vectors[None]
    distances = (points[:, None] - nearest).norm(dim=-1)
    return fraction, nearest, distances


def bone_frames(endpoints):
    """Return bone origins, lengths, and stable orthonormal frame columns."""
    origins = endpoints[:, 0]
    vectors = endpoints[:, 1] - endpoints[:, 0]
    lengths = vectors.norm(dim=-1).clamp(min=1e-8)
    tangent = vectors / lengths[:, None]
    reference = torch.zeros_like(tangent)
    reference[:, 2] = 1
    near_parallel = tangent[:, 2].abs() > 0.92
    reference[near_parallel] = tangent.new_tensor([0.0, 1.0, 0.0])
    side = torch.linalg.cross(reference, tangent, dim=-1)
    side = side / side.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    normal = torch.linalg.cross(tangent, side, dim=-1)
    basis = torch.stack([tangent, side, normal], dim=-1)
    return origins, lengths, basis
