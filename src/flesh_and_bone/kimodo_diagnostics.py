"""Anatomical diagnostics for retargeted Kimodo skeleton sequences."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class AnatomyThresholds:
    """Broad gates for gross retarget failures, expressed in display units."""

    pelvis_tilt_warn_deg: float = 20.0
    pelvis_tilt_fail_deg: float = 35.0
    socket_above_hips_warn_m: float = 0.020
    socket_above_hips_fail_m: float = 0.050
    segment_drift_warn_m: float = 0.0001
    segment_drift_fail_m: float = 0.0005
    contact_drift_warn_m: float = 0.020
    contact_drift_fail_m: float = 0.050


def _status(value: float, warning: float, failure: float) -> str:
    if value > failure:
        return "fail"
    if value > warning:
        return "warn"
    return "pass"


def _joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    left = a - b
    right = c - b
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(
        right, axis=-1
    )
    cosine = np.sum(left * right, axis=-1) / np.clip(
        denominator, 1e-12, None
    )
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _contact_spans(mask: np.ndarray, minimum: int = 3):
    edges = np.flatnonzero(
        np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    )
    return [
        (int(start), int(stop))
        for start, stop in zip(edges[::2], edges[1::2])
        if stop - start >= minimum
    ]


def _contact_drift(
    positions: np.ndarray,
    contacts: np.ndarray | None,
    channel: int,
    release_blend_frames: int = 3,
) -> tuple[float | None, int]:
    if contacts is None or contacts.ndim != 2 or channel >= contacts.shape[1]:
        return None, 0
    spans = _contact_spans(np.asarray(contacts[:, channel], dtype=bool))
    if not spans:
        return None, 0
    worst = 0.0
    for start, stop in spans:
        locked_stop = max(start + 1, stop - int(release_blend_frames))
        displacement = np.linalg.norm(
            positions[start:locked_stop] - positions[start], axis=-1
        )
        worst = max(worst, float(displacement.max(initial=0.0)))
    return worst, len(spans)


def hierarchy_segments(
    bone_endpoints: np.ndarray,
    parents: np.ndarray,
    include_count: int | None = None,
) -> np.ndarray:
    """Join parent and child heads, excluding optional helper bones."""
    endpoints = np.asarray(bone_endpoints)
    parents = np.asarray(parents, dtype=np.int64)
    squeeze = endpoints.ndim == 3
    if squeeze:
        endpoints = endpoints[None]
    if endpoints.ndim != 4 or endpoints.shape[2:] != (2, 3):
        raise ValueError("bone_endpoints must have shape [frames,bones,2,3]")
    limit = endpoints.shape[1] if include_count is None else int(include_count)
    edges = [
        (int(parent), child)
        for child, parent in enumerate(parents[:limit])
        if 0 <= int(parent) < limit
    ]
    segments = np.stack(
        [
            np.stack(
                [endpoints[:, parent, 0], endpoints[:, child, 0]], axis=1
            )
            for parent, child in edges
        ],
        axis=1,
    )
    return segments[0] if squeeze else segments


def bone_group_colors(dominant_bone: np.ndarray, bone_names) -> np.ndarray:
    """Assign readable bilateral/torso colors to dominant-bone cell labels."""
    palette = {
        "left_leg": np.array([0.93, 0.24, 0.25], dtype=np.float32),
        "right_leg": np.array([0.20, 0.52, 0.96], dtype=np.float32),
        "left_arm": np.array([0.97, 0.55, 0.16], dtype=np.float32),
        "right_arm": np.array([0.25, 0.82, 0.53], dtype=np.float32),
        "torso": np.array([0.70, 0.42, 0.91], dtype=np.float32),
        "head": np.array([0.96, 0.84, 0.35], dtype=np.float32),
        "other": np.array([0.72, 0.76, 0.80], dtype=np.float32),
    }
    lookup = []
    for name in bone_names:
        lower = str(name).lower()
        if lower.startswith("left") and any(
            token in lower for token in ("leg", "foot", "toe")
        ):
            group = "left_leg"
        elif lower.startswith("right") and any(
            token in lower for token in ("leg", "foot", "toe")
        ):
            group = "right_leg"
        elif lower.startswith("left") and any(
            token in lower for token in ("arm", "hand", "shoulder")
        ):
            group = "left_arm"
        elif lower.startswith("right") and any(
            token in lower for token in ("arm", "hand", "shoulder")
        ):
            group = "right_arm"
        elif "head" in lower or "neck" in lower:
            group = "head"
        elif "hip" in lower or "spine" in lower:
            group = "torso"
        else:
            group = "other"
        lookup.append(palette[group])
    return np.asarray(lookup)[np.asarray(dominant_bone, dtype=np.int64)]


def analyze_retarget(
    bone_endpoints: np.ndarray,
    bone_names,
    parents: np.ndarray,
    contacts: np.ndarray | None = None,
    thresholds: AnatomyThresholds | None = None,
) -> dict:
    """Measure gross anatomy errors without rejecting legitimate high motion."""
    thresholds = thresholds or AnatomyThresholds()
    endpoints = np.asarray(bone_endpoints, dtype=np.float64)
    if endpoints.ndim != 4 or endpoints.shape[2:] != (2, 3):
        raise ValueError("bone_endpoints must have shape [frames,bones,2,3]")
    names = {str(name): index for index, name in enumerate(bone_names)}
    required = {
        "Hips", "LeftUpLeg", "LeftLeg", "LeftFoot", "RightUpLeg",
        "RightLeg", "RightFoot", "LeftArm", "LeftForeArm", "LeftHand",
        "RightArm", "RightForeArm", "RightHand",
    }
    missing = sorted(required - names.keys())
    if missing:
        raise ValueError(f"missing diagnostic bones: {missing}")
    heads = endpoints[:, :, 0]

    hips = heads[:, names["Hips"]]
    left_socket = heads[:, names["LeftUpLeg"]]
    right_socket = heads[:, names["RightUpLeg"]]
    lateral = left_socket - right_socket
    lateral_norm = np.linalg.norm(lateral, axis=-1)
    pelvis_tilt = np.degrees(
        np.arcsin(
            np.clip(
                np.abs(lateral[:, 1]) / np.clip(lateral_norm, 1e-12, None),
                0.0,
                1.0,
            )
        )
    )
    socket_above = np.maximum(left_socket[:, 1], right_socket[:, 1]) - hips[:, 1]

    parent_array = np.asarray(parents, dtype=np.int64)
    segment_lengths = []
    for child, parent in enumerate(parent_array):
        if parent >= 0:
            segment_lengths.append(
                np.linalg.norm(heads[:, child] - heads[:, parent], axis=-1)
            )
    segment_lengths = np.stack(segment_lengths, axis=1)
    segment_drift = np.max(
        np.abs(segment_lengths - segment_lengths[:1]), axis=0
    )
    maximum_segment_drift = float(segment_drift.max(initial=0.0))

    angle_chains = {
        "left_knee": ("LeftUpLeg", "LeftLeg", "LeftFoot"),
        "right_knee": ("RightUpLeg", "RightLeg", "RightFoot"),
        "left_elbow": ("LeftArm", "LeftForeArm", "LeftHand"),
        "right_elbow": ("RightArm", "RightForeArm", "RightHand"),
    }
    angles = {}
    for label, chain in angle_chains.items():
        value = _joint_angle(*(heads[:, names[name]] for name in chain))
        angles[label] = {
            "minimum_deg": float(value.min()),
            "maximum_deg": float(value.max()),
        }

    left_drift, left_spans = _contact_drift(
        heads[:, names["LeftFoot"]], contacts, 0
    )
    right_drift, right_spans = _contact_drift(
        heads[:, names["RightFoot"]], contacts, 3
    )
    available_drifts = [
        value for value in (left_drift, right_drift) if value is not None
    ]
    maximum_contact_drift = max(available_drifts, default=None)

    finite = bool(np.isfinite(endpoints).all())
    metric_rows = []

    def add_metric(identifier, label, value, unit, status, detail):
        metric_rows.append(
            {
                "id": identifier,
                "label": label,
                "value": None if value is None else float(value),
                "unit": unit,
                "status": status,
                "detail": detail,
            }
        )

    tilt_max = float(pelvis_tilt.max())
    add_metric(
        "pelvis_tilt_max",
        "Maximum pelvis tilt",
        tilt_max,
        "deg",
        _status(
            tilt_max,
            thresholds.pelvis_tilt_warn_deg,
            thresholds.pelvis_tilt_fail_deg,
        ),
        "Tilt of the left-to-right hip socket axis away from horizontal.",
    )
    socket_max = float(socket_above.max())
    add_metric(
        "hip_socket_above_hips",
        "Hip socket above Hips",
        socket_max,
        "m",
        _status(
            socket_max,
            thresholds.socket_above_hips_warn_m,
            thresholds.socket_above_hips_fail_m,
        ),
        "Largest vertical excess of either upper-leg socket above Hips.",
    )
    add_metric(
        "segment_length_drift",
        "Hierarchy length drift",
        maximum_segment_drift,
        "m",
        _status(
            maximum_segment_drift,
            thresholds.segment_drift_warn_m,
            thresholds.segment_drift_fail_m,
        ),
        "Maximum parent-child distance change from frame zero.",
    )
    contact_status = "unavailable"
    if maximum_contact_drift is not None:
        contact_status = _status(
            maximum_contact_drift,
            thresholds.contact_drift_warn_m,
            thresholds.contact_drift_fail_m,
        )
    add_metric(
        "contact_foot_drift",
        "Planted-foot drift",
        maximum_contact_drift,
        "m",
        contact_status,
        (
            f"Maximum locked-core drift across {left_spans + right_spans} "
            "heel-contact spans; three-frame release blends excluded."
        ),
    )
    add_metric(
        "finite",
        "Finite skeleton values",
        1.0 if finite else 0.0,
        "bool",
        "pass" if finite else "fail",
        "All retargeted endpoints must be finite.",
    )

    severity = {"unavailable": 0, "pass": 0, "warn": 1, "fail": 2}
    verdict = max(metric_rows, key=lambda row: severity[row["status"]])["status"]
    if verdict == "unavailable":
        verdict = "pass"
    reasons = [
        row["label"] for row in metric_rows if row["status"] == verdict
    ] if verdict != "pass" else []
    worst_frame = int(np.argmax(pelvis_tilt))
    return {
        "verdict": verdict,
        "reasons": reasons,
        "worst_frame": worst_frame,
        "thresholds": asdict(thresholds),
        "metrics": metric_rows,
        "angles": angles,
        "context": {
            "frame_count": int(endpoints.shape[0]),
            "pelvis_tilt_p95_deg": float(np.quantile(pelvis_tilt, 0.95)),
            "pelvis_tilt_mean_deg": float(pelvis_tilt.mean()),
            "hip_width_min_m": float(lateral_norm.min()),
            "hip_width_max_m": float(lateral_norm.max()),
            "socket_above_hips_min_m": float(socket_above.min()),
            "root_vertical_range_m": float(np.ptp(hips[:, 1])),
        },
        "per_frame": {
            "pelvis_tilt_deg": [float(value) for value in pelvis_tilt],
            "socket_above_hips_m": [float(value) for value in socket_above],
        },
    }
