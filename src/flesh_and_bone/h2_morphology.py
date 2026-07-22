"""Nonuniform H2 tissue envelope and wound-region specification."""

from dataclasses import dataclass

import torch

from .morphology import BodyPlan, checker_at_points, embed_points
from .skeleton import HScaffold, segment_projection


REGION_NAMES = (
    "left_upper",
    "left_lower",
    "thin_bridge",
    "right_upper",
    "right_lower",
    "terminal_bulb",
    "off_axis_pad",
)


@dataclass(frozen=True)
class H2BodyPlan(BodyPlan):
    """Body plan plus anatomical regions and a predeclared wound patch."""

    region: torch.Tensor
    wound_mask: torch.Tensor
    reference_bone_distance: torch.Tensor
    uniform_density: float
    region_names: tuple[str, ...]


def _inside_ellipsoid(points, center, axes):
    return (((points - center[None]) / axes[None]).square().sum(dim=1) <= 1)


def build_h2_body_plan(scaffold=None, spacing=0.14, checker_size=0.28,
                       weight_sigma=0.20, device=None,
                       dtype=torch.float32):
    """Sample an asymmetric envelope whose anatomy is not a bone offset."""
    scaffold = scaffold or HScaffold()
    rest = scaffold.frame(0.0, device=device, dtype=dtype)
    maximum_extent = 0.43
    low = rest.endpoints.amin(dim=(0, 1)) - maximum_extent
    high = rest.endpoints.amax(dim=(0, 1)) + maximum_extent
    high[0] += 0.20
    axes = [
        torch.arange(low[axis], high[axis] + 0.5 * spacing, spacing,
                     device=device, dtype=dtype)
        for axis in range(3)
    ]
    x, y, z = torch.meshgrid(*axes, indexing="ij")
    candidates = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
    fraction, _, distances = segment_projection(candidates, rest.endpoints)

    radii = torch.stack([
        0.19 + 0.12 * fraction[:, 0],
        fraction[:, 1].new_full(fraction[:, 1].shape, 0.18),
        fraction[:, 2].new_full(fraction[:, 2].shape, 0.12),
        fraction[:, 3].new_full(fraction[:, 3].shape, 0.15),
        0.18 + 0.08 * fraction[:, 4],
    ], dim=1)
    variable_capsules = (distances <= radii).any(dim=1)

    bulb_center = rest.endpoints[0, 1] + candidates.new_tensor([0.0, 0.02, 0.0])
    bulb_axes = candidates.new_tensor([0.38, 0.32, 0.23])
    pad_center = rest.endpoints[2, 1] + candidates.new_tensor([0.26, 0.18, 0.0])
    pad_axes = candidates.new_tensor([0.32, 0.23, 0.17])
    bulb = _inside_ellipsoid(candidates, bulb_center, bulb_axes)
    pad = _inside_ellipsoid(candidates, pad_center, pad_axes)
    sites = candidates[variable_capsules | bulb | pad]

    weights, local = embed_points(sites, rest, weight_sigma)
    dominant = weights.argmax(dim=1)
    checker = checker_at_points(sites, low, checker_size)
    _, _, site_bone_distances = segment_projection(sites, rest.endpoints)
    reference_bone_distance = site_bone_distances.amin(dim=1)

    site_bulb = _inside_ellipsoid(sites, bulb_center, bulb_axes) & (
        sites[:, 1] > 0.58
    )
    site_pad = _inside_ellipsoid(sites, pad_center, pad_axes) & (
        sites[:, 0] > rest.endpoints[2, 1, 0]
    )
    region = dominant.clone()
    region[site_bulb] = 5
    region[site_pad] = 6
    wound_mask = site_bulb & (sites[:, 1] > 0.84)

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
    return H2BodyPlan(
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
        radius=0.24,
        region=region,
        wound_mask=wound_mask,
        reference_bone_distance=reference_bone_distance,
        uniform_density=float(target_density.median().item()),
        region_names=REGION_NAMES,
    )
