import json
import struct

import numpy as np
import torch

from flesh_and_bone.runtime_export import (
    BODY_HEADER,
    BODY_MAGIC,
    BODY_VERSION,
    MODEL_HEADER,
    MODEL_MAGIC,
    _bone_source_anchors,
    _six_neighbors,
    _top_six,
    export_runtime_model,
)


def test_six_neighbor_export_has_fixed_aligned_width():
    points = np.asarray([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ])
    neighbors = _six_neighbors(points, 1.0)

    assert neighbors.shape == (3, 8)
    assert neighbors[0, :2].tolist() == [1, -1]
    assert set(neighbors[1, :2].tolist()) == {0, 2}
    assert np.all(neighbors[:, 2:] == -1)


def test_top_six_export_normalizes_and_pads():
    source = np.arange(24, dtype=np.float32)[None] + 1
    indices, weights = _top_six(source)

    assert indices.shape == (1, 8)
    assert weights.shape == (1, 8)
    assert indices[0, :6].tolist() == [23, 22, 21, 20, 19, 18]
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights[0, 6:] == 0)


def test_source_anchors_project_onto_dominant_bone_segments():
    points = np.asarray([
        [0.5, 1.0, 0.0],
        [2.0, 0.5, 0.0],
        [-1.0, 4.0, 0.0],
    ], dtype=np.float32)
    indices = np.zeros((3, 8), dtype=np.uint16)
    indices[:, 0] = [0, 1, 0]
    endpoints = np.asarray([
        [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
    ], dtype=np.float32)

    anchors = _bone_source_anchors(points, indices, endpoints)

    assert np.allclose(anchors[0], [0.0, 1.0, 0.0])
    assert np.allclose(anchors[1], [2.0, 0.0, 0.0])
    assert np.allclose(anchors[2], [0.0, 2.0, 0.0])


def test_model_export_preserves_declared_binary_contract(tmp_path):
    state = {
        "coefficient_maxima": torch.tensor([1440.0, 432.0]),
        "network.0.weight": torch.arange(160).reshape(32, 5).float(),
        "network.0.bias": torch.arange(32).float(),
        "network.2.weight": torch.arange(1024).reshape(32, 32).float(),
        "network.2.bias": torch.arange(32).float(),
        "network.4.weight": torch.arange(64).reshape(2, 32).float(),
        "network.4.bias": torch.arange(2).float(),
    }
    checkpoint = tmp_path / "checkpoint.pt"
    metrics = tmp_path / "metrics.json"
    output = tmp_path / "model.fnm"
    torch.save(state, checkpoint)
    metrics.write_text(json.dumps({
        "backbone": {"coefficients": [1.0, 0.44, 1200.0, 0.0, 1.0]}
    }))

    report = export_runtime_model(checkpoint, metrics, output)
    payload = output.read_bytes()
    header = MODEL_HEADER.unpack_from(payload)

    assert header[0] == MODEL_MAGIC
    assert header[2:4] == (32, 5)
    assert report["learned_parameters"] == 1314
    assert len(payload) == MODEL_HEADER.size + (5 + 2 + 1314) * 4
    assert BODY_HEADER.size == struct.calcsize("<4sIIIIfff")
    assert BODY_MAGIC == b"FNB1"
    assert BODY_VERSION == 2
