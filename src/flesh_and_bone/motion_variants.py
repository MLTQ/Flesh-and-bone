"""Deterministic periodic motion transformations for H6M controls."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MotionCycle:
    """One named LBS/bone-endpoint cycle with shared phase semantics."""

    name: str
    description: str
    lbs_positions: torch.Tensor
    bone_endpoints: torch.Tensor


def periodic_catmull_rom(values, output_frames):
    """Resample a closed uniformly sampled cycle without a linear-knot impulse."""
    if values.ndim < 1 or values.shape[0] < 3:
        raise ValueError("periodic input needs at least three frames")
    output_frames = int(output_frames)
    if output_frames < 3:
        raise ValueError("periodic output needs at least three frames")
    source_frames = values.shape[0]
    phase = (
        torch.arange(output_frames, device=values.device, dtype=values.dtype)
        * (source_frames / output_frames)
    )
    left = torch.floor(phase).to(torch.long)
    fraction = phase - left.to(phase.dtype)
    shape = (output_frames,) + (1,) * (values.ndim - 1)
    t = fraction.reshape(shape)
    p0 = values[(left - 1).remainder(source_frames)]
    p1 = values[left.remainder(source_frames)]
    p2 = values[(left + 1).remainder(source_frames)]
    p3 = values[(left + 2).remainder(source_frames)]
    return 0.5 * (
        2 * p1
        + (-p0 + p2) * t
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t.square()
        + (-p0 + 3 * p1 - 3 * p2 + p3) * t.pow(3)
    )


def reverse_cycle(values):
    """Reverse time while retaining phase zero as the cycle origin."""
    indices = torch.cat([
        torch.zeros(1, device=values.device, dtype=torch.long),
        torch.arange(
            values.shape[0] - 1,
            0,
            -1,
            device=values.device,
            dtype=torch.long,
        ),
    ])
    return values[indices]


def walk_then_hold(values, hold_frames=15):
    """Append a phase-zero dwell to one traversal of a periodic cycle."""
    hold_frames = int(hold_frames)
    if hold_frames < 1:
        raise ValueError("hold_frames must be positive")
    hold = values[:1].expand((hold_frames,) + values.shape[1:])
    return torch.cat([values, hold], dim=0)


def controlled_motion_cycles(lbs_positions, bone_endpoints):
    """Build the frozen H6M calibration and four predeclared motion arms."""
    if lbs_positions.shape[0] != bone_endpoints.shape[0]:
        raise ValueError("LBS and bone cycles need the same phase count")

    def cycle(name, description, transform):
        return MotionCycle(
            name=name,
            description=description,
            lbs_positions=transform(lbs_positions),
            bone_endpoints=transform(bone_endpoints),
        )

    source_frames = lbs_positions.shape[0]
    return (
        MotionCycle(
            name="walk_replay",
            description="original 29-phase H5U walk calibration",
            lbs_positions=lbs_positions,
            bone_endpoints=bone_endpoints,
        ),
        cycle(
            "reverse",
            "phase-zero-anchored reverse walk",
            reverse_cycle,
        ),
        cycle(
            "half_speed",
            "periodic cubic resampling to 58 phases",
            lambda value: periodic_catmull_rom(value, 2 * source_frames),
        ),
        cycle(
            "fast_1p526",
            "periodic cubic resampling to 19 phases",
            lambda value: periodic_catmull_rom(
                value, round(source_frames / 1.5)
            ),
        ),
        cycle(
            "walk_then_hold",
            "one walk traversal followed by a 15-frame phase-zero dwell",
            lambda value: walk_then_hold(value, 15),
        ),
    )
