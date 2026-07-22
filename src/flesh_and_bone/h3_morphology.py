"""Fine humanoid tissue envelope with extra-skeletal lobes."""

from dataclasses import dataclass

import torch

from .humanoid_skeleton import HumanoidScaffold
from .morphology import BodyPlan, checker_at_points, embed_points
from .skeleton import segment_projection


H3_REGION_NAMES = (
    "torso",
    "neck",
    "head",
    "left_cheek",
    "right_cheek",
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
    "pelvis",
    "left_buttock",
    "right_buttock",
    "left_thigh",
    "left_lower_leg",
    "right_thigh",
    "right_lower_leg",
    "left_foot",
    "right_foot",
)

CRITICAL_REGION_INDICES = (3, 4, 10, 11)


@dataclass(frozen=True)
class H3BodyPlan(BodyPlan):
    """Humanoid reference field plus learned-fate experiment annotations."""

    region: torch.Tensor
    region_names: tuple[str, ...]
    critical_region_indices: tuple[int, ...]
    wound_mask: torch.Tensor
    reference_bone_distance: torch.Tensor
    uniform_density: float
    splat_scale: torch.Tensor


def _inside_ellipsoid(points, center, axes):
    return (((points - center[None]) / axes[None]).square().sum(dim=1) <= 1)


def build_h3_body_plan(scaffold=None, spacing=0.10, checker_size=0.20,
                       weight_sigma=0.18, device=None,
                       dtype=torch.float32):
    """Sample a fine humanoid envelope not reducible to bone capsules."""
    scaffold = scaffold or HumanoidScaffold()
    rest = scaffold.frame(0.0, device=device, dtype=dtype)
    low = rest.endpoints.amin(dim=(0, 1)) + rest.endpoints.new_tensor(
        [-0.48, -0.34, -0.52]
    )
    high = rest.endpoints.amax(dim=(0, 1)) + rest.endpoints.new_tensor(
        [0.48, 0.44, 0.52]
    )
    axes = [
        torch.arange(low[axis], high[axis] + 0.5 * spacing, spacing,
                     device=device, dtype=dtype)
        for axis in range(3)
    ]
    x, y, z = torch.meshgrid(*axes, indexing="ij")
    candidates = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
    fraction, _, distances = segment_projection(candidates, rest.endpoints)

    radii = torch.stack([
        fraction[:, 0].new_full(fraction[:, 0].shape, 0.22),
        fraction[:, 1].new_full(fraction[:, 1].shape, 0.14),
        fraction[:, 2].new_full(fraction[:, 2].shape, 0.15),
        fraction[:, 3].new_full(fraction[:, 3].shape, 0.15),
        0.16 - 0.035 * fraction[:, 4],
        0.13 - 0.020 * fraction[:, 5],
        fraction[:, 6].new_full(fraction[:, 6].shape, 0.15),
        0.16 - 0.035 * fraction[:, 7],
        0.13 - 0.020 * fraction[:, 8],
        fraction[:, 9].new_full(fraction[:, 9].shape, 0.18),
        0.21 - 0.050 * fraction[:, 10],
        0.16 - 0.035 * fraction[:, 11],
        fraction[:, 12].new_full(fraction[:, 12].shape, 0.18),
        0.21 - 0.050 * fraction[:, 13],
        0.16 - 0.035 * fraction[:, 14],
    ], dim=1)
    limb_capsules = (distances <= radii).any(dim=1)

    def ellipsoid(center, shape):
        return _inside_ellipsoid(
            candidates,
            candidates.new_tensor(center),
            candidates.new_tensor(shape),
        )

    torso = ellipsoid((0.0, 0.28, 0.0), (0.45, 0.62, 0.25))
    chest = ellipsoid((0.0, 0.57, 0.0), (0.51, 0.32, 0.28))
    pelvis = ellipsoid((0.0, -0.23, -0.01), (0.47, 0.31, 0.30))
    head = ellipsoid((0.0, 1.29, 0.02), (0.34, 0.41, 0.31))
    left_cheek = ellipsoid((-0.19, 1.21, 0.22), (0.19, 0.17, 0.16))
    right_cheek = ellipsoid((0.19, 1.21, 0.22), (0.19, 0.17, 0.16))
    left_buttock = ellipsoid((-0.21, -0.29, -0.23), (0.28, 0.25, 0.21))
    right_buttock = ellipsoid((0.21, -0.29, -0.23), (0.28, 0.25, 0.21))
    left_hand = ellipsoid((-0.70, -0.35, 0.0), (0.15, 0.19, 0.13))
    right_hand = ellipsoid((0.70, -0.35, 0.0), (0.15, 0.19, 0.13))
    left_foot = ellipsoid((-0.25, -1.64, 0.15), (0.18, 0.14, 0.29))
    right_foot = ellipsoid((0.25, -1.64, 0.15), (0.18, 0.14, 0.29))
    envelope = (
        limb_capsules | torso | chest | pelvis | head
        | left_cheek | right_cheek | left_buttock | right_buttock
        | left_hand | right_hand | left_foot | right_foot
    )
    sites = candidates[envelope]

    weights, local = embed_points(sites, rest, weight_sigma)
    dominant = weights.argmax(dim=1)
    checker = checker_at_points(sites, low, checker_size)
    _, _, site_bone_distances = segment_projection(sites, rest.endpoints)
    reference_bone_distance = site_bone_distances.amin(dim=1)

    region_by_bone = sites.new_tensor([
        0, 1, 2, 5, 5, 5, 6, 6, 6,
        9, 12, 13, 9, 14, 15,
    ], dtype=torch.long)
    region = region_by_bone[dominant]

    def site_ellipsoid(center, shape):
        return _inside_ellipsoid(
            sites, sites.new_tensor(center), sites.new_tensor(shape)
        )

    site_left_hand = site_ellipsoid((-0.70, -0.35, 0.0), (0.15, 0.19, 0.13))
    site_right_hand = site_ellipsoid((0.70, -0.35, 0.0), (0.15, 0.19, 0.13))
    site_left_foot = site_ellipsoid((-0.25, -1.64, 0.15), (0.18, 0.14, 0.29))
    site_right_foot = site_ellipsoid((0.25, -1.64, 0.15), (0.18, 0.14, 0.29))
    site_left_buttock = site_ellipsoid((-0.21, -0.29, -0.23), (0.28, 0.25, 0.21))
    site_right_buttock = site_ellipsoid((0.21, -0.29, -0.23), (0.28, 0.25, 0.21))
    site_left_cheek = site_ellipsoid((-0.19, 1.21, 0.22), (0.19, 0.17, 0.16))
    site_right_cheek = site_ellipsoid((0.19, 1.21, 0.22), (0.19, 0.17, 0.16))
    region[site_left_hand] = 7
    region[site_right_hand] = 8
    region[site_left_foot] = 16
    region[site_right_foot] = 17
    region[site_left_buttock & (sites[:, 0] <= 0)] = 10
    region[site_right_buttock & (sites[:, 0] > 0)] = 11
    region[site_left_cheek & (sites[:, 0] <= 0)] = 3
    region[site_right_cheek & (sites[:, 0] > 0)] = 4

    wound_mask = (
        (region == 4)
        & (sites[:, 2] > 0.17)
        & (sites[:, 1] > 1.15)
    )
    material_scale = sites.new_tensor([
        1.15, 0.80, 0.95, 1.25, 1.25, 0.86, 0.86, 0.95, 0.95,
        1.15, 1.30, 1.30, 1.00, 0.80, 1.00, 0.80, 0.92, 0.92,
    ])
    splat_scale = material_scale[region]

    density_radius = 1.9 * spacing
    pair_distance = torch.cdist(sites, sites)
    target_density = torch.exp(-(pair_distance / density_radius).square())
    target_density.fill_diagonal_(0)
    target_density = target_density.sum(dim=1)

    gap_reference = sites.new_tensor([
        [-0.52, 0.40, 0.0],
        [0.52, 0.40, 0.0],
    ])
    gap_weights, gap_local = embed_points(gap_reference, rest, weight_sigma)
    return H3BodyPlan(
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
        radius=0.31,
        region=region,
        region_names=H3_REGION_NAMES,
        critical_region_indices=CRITICAL_REGION_INDICES,
        wound_mask=wound_mask,
        reference_bone_distance=reference_bone_distance,
        uniform_density=float(target_density.median().item()),
        splat_scale=splat_scale,
    )
