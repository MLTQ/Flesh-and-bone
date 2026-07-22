"""CPU-fast feature-dataset layout contract for H5."""

from types import SimpleNamespace

import torch

from flesh_and_bone.flesh_rollout import teacher_dataset


def test_teacher_dataset_preserves_phase_labels_and_example_count():
    shape = (2, 3, 4, 3)
    vector = torch.zeros(shape)
    trajectory = SimpleNamespace(
        residual=vector,
        velocity=vector,
        acceleration=vector,
        neighbor_residual=vector,
        neighbor_velocity=vector,
        lbs_acceleration=torch.zeros(2, 4, 3),
        stiffness=torch.ones(4),
    )
    volume = SimpleNamespace(bone_distance=torch.zeros(4))
    features, targets, phase = teacher_dataset(trajectory, volume)
    assert features.shape == (24, 17)
    assert targets.shape == (24, 3)
    assert phase.tolist() == [0] * 12 + [1] * 12
