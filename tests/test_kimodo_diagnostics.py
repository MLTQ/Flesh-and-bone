import numpy as np

from flesh_and_bone.kimodo_diagnostics import (
    analyze_retarget,
    bone_group_colors,
    hierarchy_segments,
)


NAMES = (
    "Hips",
    "LeftUpLeg", "LeftLeg", "LeftFoot",
    "RightUpLeg", "RightLeg", "RightFoot",
    "LeftArm", "LeftForeArm", "LeftHand",
    "RightArm", "RightForeArm", "RightHand",
    "Head", "head_end",
)
PARENTS = np.array(
    [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13],
    dtype=np.int64,
)


def _sequence(frames=8):
    heads = np.array(
        [
            [0.00, 1.00, 0.00],
            [0.11, 0.91, 0.00], [0.11, 0.52, 0.01], [0.11, 0.12, 0.03],
            [-0.11, 0.91, 0.00], [-0.11, 0.52, 0.01], [-0.11, 0.12, 0.03],
            [0.22, 1.42, 0.00], [0.49, 1.30, 0.00], [0.70, 1.20, 0.00],
            [-0.22, 1.42, 0.00], [-0.49, 1.30, 0.00], [-0.70, 1.20, 0.00],
            [0.00, 1.63, 0.00], [0.00, 1.77, 0.00],
        ],
        dtype=np.float64,
    )
    sequence = np.repeat(heads[None], frames, axis=0)
    tails = sequence + np.array([0.0, 0.04, 0.0])
    return np.stack([sequence, tails], axis=2)


def test_normal_skeleton_passes_gross_anatomy_screen():
    result = analyze_retarget(_sequence(), NAMES, PARENTS)

    assert result["verdict"] == "pass"
    assert result["worst_frame"] == 0
    assert next(
        row for row in result["metrics"] if row["id"] == "pelvis_tilt_max"
    )["value"] == 0.0


def test_tilted_pelvis_and_raised_socket_fail():
    endpoints = _sequence()
    endpoints[:, 1, :, 1] += 0.207
    endpoints[:, 4, :, 1] -= 0.117

    result = analyze_retarget(endpoints, NAMES, PARENTS)
    metrics = {row["id"]: row for row in result["metrics"]}

    assert result["verdict"] == "fail"
    assert metrics["pelvis_tilt_max"]["value"] > 35.0
    assert metrics["hip_socket_above_hips"]["value"] > 0.05


def test_contact_drift_is_only_gated_when_contact_spans_exist():
    endpoints = _sequence()
    endpoints[:, 3, :, 0] += np.linspace(0.0, 0.12, endpoints.shape[0])[:, None]
    contacts = np.zeros((endpoints.shape[0], 4), dtype=bool)
    contacts[:, 0] = True

    result = analyze_retarget(endpoints, NAMES, PARENTS, contacts=contacts)
    contact = next(
        row for row in result["metrics"] if row["id"] == "contact_foot_drift"
    )

    assert contact["status"] == "fail"
    assert contact["value"] > 0.05


def test_hierarchy_segments_excludes_helpers_and_colors_sides():
    endpoints = _sequence(frames=1)[0]
    segments = hierarchy_segments(endpoints, PARENTS, include_count=14)
    colors = bone_group_colors(np.array([1, 4, 7, 10, 13]), NAMES)

    assert segments.shape == (13, 2, 3)
    assert not np.allclose(colors[0], colors[1])
    assert not np.allclose(colors[2], colors[3])
    assert not np.allclose(colors[4], colors[0])
