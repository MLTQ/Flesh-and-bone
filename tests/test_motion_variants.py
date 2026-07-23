"""CPU-fast contracts for H6M periodic motion transformations."""

import torch

from flesh_and_bone.motion_variants import (
    controlled_motion_cycles,
    periodic_catmull_rom,
    reverse_cycle,
    walk_then_hold,
)


def test_reverse_retains_phase_zero_and_reverses_remaining_frames():
    values = torch.arange(5, dtype=torch.float32)[:, None]
    assert reverse_cycle(values)[:, 0].tolist() == [0, 4, 3, 2, 1]


def test_periodic_cubic_upsample_reproduces_source_knots():
    values = torch.tensor([0.0, 1.0, 0.0, -1.0])[:, None]
    upsampled = periodic_catmull_rom(values, 8)
    assert torch.allclose(upsampled[::2], values)


def test_controlled_cycles_have_frozen_names_counts_and_shared_phases():
    lbs = torch.zeros(29, 3, 3)
    bones = torch.zeros(29, 2, 2, 3)
    cycles = controlled_motion_cycles(lbs, bones)
    assert [cycle.name for cycle in cycles] == [
        "walk_replay",
        "reverse",
        "half_speed",
        "fast_1p526",
        "walk_then_hold",
    ]
    assert [cycle.lbs_positions.shape[0] for cycle in cycles] == [
        29, 29, 58, 19, 44
    ]
    assert all(
        cycle.lbs_positions.shape[0] == cycle.bone_endpoints.shape[0]
        for cycle in cycles
    )
    assert walk_then_hold(lbs, 3).shape[0] == 32
