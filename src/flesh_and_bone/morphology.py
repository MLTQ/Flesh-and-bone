"""Bone-conditioned developmental niches for the H body plan."""

from dataclasses import dataclass

import torch

from .skeleton import HScaffold, ScaffoldFrame, bone_frames, segment_projection


@dataclass(frozen=True)
class BodyPlan:
    """Canonical tissue sites and their persistent skeletal coordinates."""

    reference_sites: torch.Tensor
    bone_weights: torch.Tensor
    bone_local: torch.Tensor
    dominant_bone: torch.Tensor
    checker: torch.Tensor
    target_density: torch.Tensor
    gap_reference: torch.Tensor
    gap_weights: torch.Tensor
    gap_local: torch.Tensor
    checker_origin: torch.Tensor
    checker_size: float
    weight_sigma: float
    spacing: float
    radius: float

    @property
    def site_count(self):
        return self.reference_sites.shape[0]


def checker_at_points(points, origin, checker_size):
    """Evaluate the planar checker skin extruded through tissue thickness."""
    coordinate = torch.floor(
        (points - origin[None]) / checker_size
    ).to(torch.long)
    return coordinate[:, :2].sum(dim=1).remainder(2)


def embed_points(points, frame, weight_sigma, top_bones=3):
    """Embed arbitrary world points in soft coordinates of one scaffold frame."""
    endpoints = frame.endpoints
    fraction, nearest, distances = segment_projection(points, endpoints)
    count, bones = distances.shape
    selected = distances.topk(min(top_bones, bones), largest=False, dim=1).indices
    logits = torch.full_like(distances, -torch.inf)
    selected_logits = -distances.gather(1, selected).square() / (2 * weight_sigma ** 2)
    logits.scatter_(1, selected, selected_logits)
    weights = torch.softmax(logits, dim=1)

    origins, lengths, basis = bone_frames(endpoints)
    offset = points[:, None] - nearest
    side = (offset * basis[None, :, :, 1]).sum(dim=-1)
    normal = (offset * basis[None, :, :, 2]).sum(dim=-1)
    # Store an unclamped axial coordinate. Capsule distance uses the clamped
    # projection above, but an unclamped coordinate is required for every soft
    # bone candidate to reconstruct the exact rest point, including points just
    # beyond a segment endpoint.
    axial = (
        (points[:, None] - origins[None]) * basis[None, :, :, 0]
    ).sum(dim=-1) / lengths[None]
    local = torch.stack([axial, side, normal], dim=-1)
    return weights, local


def build_h_body_plan(scaffold=None, spacing=0.14, radius=0.24,
                      checker_size=0.28, weight_sigma=0.20,
                      device=None, dtype=torch.float32):
    """Voxel-sample the union of five capsules into persistent splat niches."""
    scaffold = scaffold or HScaffold()
    rest = scaffold.frame(0.0, device=device, dtype=dtype)
    low = rest.endpoints.amin(dim=(0, 1)) - radius
    high = rest.endpoints.amax(dim=(0, 1)) + radius
    axes = [
        torch.arange(low[axis], high[axis] + 0.5 * spacing, spacing,
                     device=device, dtype=dtype)
        for axis in range(3)
    ]
    x, y, z = torch.meshgrid(*axes, indexing="ij")
    candidates = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
    _, _, distances = segment_projection(candidates, rest.endpoints)
    sites = candidates[distances.amin(dim=1) <= radius]
    weights, local = embed_points(sites, rest, weight_sigma)
    dominant = weights.argmax(dim=1)

    checker = checker_at_points(sites, low, checker_size)

    density_radius = 1.9 * spacing
    pair_distance = torch.cdist(sites, sites)
    target_density = torch.exp(-(pair_distance / density_radius).square())
    target_density.fill_diagonal_(0)
    target_density = target_density.sum(dim=1)

    gap_reference = sites.new_tensor([
        [0.0, 0.46, 0.0],
        [0.0, -0.46, 0.0],
    ])
    gap_weights, gap_local = embed_points(gap_reference, rest, weight_sigma)
    return BodyPlan(
        reference_sites=sites,
        bone_weights=weights,
        bone_local=local,
        dominant_bone=dominant,
        checker=checker,
        target_density=target_density,
        gap_reference=gap_reference,
        gap_weights=gap_weights,
        gap_local=gap_local,
        checker_origin=low,
        checker_size=float(checker_size),
        weight_sigma=float(weight_sigma),
        spacing=float(spacing),
        radius=float(radius),
    )


def deform_embedded(weights, local, frame):
    """Linear-blend persistent bone-local coordinates into one skeleton frame."""
    origins, lengths, basis = bone_frames(frame.endpoints)
    candidate = (
        origins[None]
        + local[..., 0:1] * lengths[None, :, None] * basis[None, :, :, 0]
        + local[..., 1:2] * basis[None, :, :, 1]
        + local[..., 2:3] * basis[None, :, :, 2]
    )
    return (candidate * weights[..., None]).sum(dim=1)


def deform_body_plan(body_plan, frame):
    """Return moving niche positions and negative-space probe positions."""
    sites = deform_embedded(body_plan.bone_weights, body_plan.bone_local, frame)
    gaps = deform_embedded(body_plan.gap_weights, body_plan.gap_local, frame)
    return sites, gaps
