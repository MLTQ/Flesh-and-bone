"""Fifteen-bone humanoid precursor with a bounded walk/sway cycle."""

import torch

from .skeleton import ScaffoldFrame, _rotation_x, _rotation_z


HUMANOID_BONE_NAMES = (
    "trunk",
    "neck",
    "head_axis",
    "left_clavicle",
    "left_upper_arm",
    "left_lower_arm",
    "right_clavicle",
    "right_upper_arm",
    "right_lower_arm",
    "left_hip",
    "left_thigh",
    "left_lower_leg",
    "right_hip",
    "right_thigh",
    "right_lower_leg",
)


class HumanoidScaffold:
    """Connected humanoid graph used to test non-bone-offset flesh."""

    bone_names = HUMANOID_BONE_NAMES

    @property
    def bone_count(self):
        return len(self.bone_names)

    def frame(self, time=0.0, device=None, dtype=torch.float32):
        """Return a continuous frame whose zero-time pose is the assembly pose."""
        time = torch.as_tensor(time, device=device, dtype=dtype)
        zero = time.new_zeros(())
        walk = torch.sin(time)
        counter = -walk

        pelvis = torch.stack([zero, time.new_tensor(-0.15), zero])
        chest = torch.stack([zero, time.new_tensor(0.62), zero])
        neck = torch.stack([zero, time.new_tensor(0.94), zero])
        head = torch.stack([zero, time.new_tensor(1.28), zero])
        left_shoulder = torch.stack([
            time.new_tensor(-0.43), time.new_tensor(0.64), zero
        ])
        right_shoulder = torch.stack([
            time.new_tensor(0.43), time.new_tensor(0.64), zero
        ])
        left_hip = torch.stack([
            time.new_tensor(-0.23), time.new_tensor(-0.18), zero
        ])
        right_hip = torch.stack([
            time.new_tensor(0.23), time.new_tensor(-0.18), zero
        ])

        left_elbow = left_shoulder + torch.stack([
            time.new_tensor(-0.16), time.new_tensor(-0.50), -0.18 * walk
        ])
        left_hand = left_elbow + torch.stack([
            time.new_tensor(-0.11), time.new_tensor(-0.47), -0.15 * walk
        ])
        right_elbow = right_shoulder + torch.stack([
            time.new_tensor(0.16), time.new_tensor(-0.50), -0.18 * counter
        ])
        right_hand = right_elbow + torch.stack([
            time.new_tensor(0.11), time.new_tensor(-0.47), -0.15 * counter
        ])

        left_knee = left_hip + torch.stack([
            time.new_tensor(-0.03), time.new_tensor(-0.72), 0.22 * walk
        ])
        left_ankle = left_knee + torch.stack([
            time.new_tensor(0.01),
            time.new_tensor(-0.69) + 0.035 * (1 - torch.cos(time)),
            0.14 * walk + 0.06 * (1 - torch.cos(time)),
        ])
        right_knee = right_hip + torch.stack([
            time.new_tensor(0.03), time.new_tensor(-0.72), 0.22 * counter
        ])
        right_ankle = right_knee + torch.stack([
            time.new_tensor(-0.01),
            time.new_tensor(-0.69) + 0.035 * (1 - torch.cos(time)),
            0.14 * counter + 0.06 * (1 - torch.cos(time)),
        ])

        endpoints = torch.stack([
            torch.stack([pelvis, chest]),
            torch.stack([chest, neck]),
            torch.stack([neck, head]),
            torch.stack([chest, left_shoulder]),
            torch.stack([left_shoulder, left_elbow]),
            torch.stack([left_elbow, left_hand]),
            torch.stack([chest, right_shoulder]),
            torch.stack([right_shoulder, right_elbow]),
            torch.stack([right_elbow, right_hand]),
            torch.stack([pelvis, left_hip]),
            torch.stack([left_hip, left_knee]),
            torch.stack([left_knee, left_ankle]),
            torch.stack([pelvis, right_hip]),
            torch.stack([right_hip, right_knee]),
            torch.stack([right_knee, right_ankle]),
        ])
        global_rotation = _rotation_z(0.055 * torch.sin(0.47 * time)) @ (
            _rotation_x(0.045 * torch.sin(0.61 * time))
        )
        endpoints = endpoints @ global_rotation.T
        root_translation = torch.stack([
            0.045 * torch.sin(0.47 * time),
            0.018 * (1 - torch.cos(0.94 * time)),
            time.new_zeros(()),
        ])
        endpoints = endpoints + root_translation
        return ScaffoldFrame(endpoints)
