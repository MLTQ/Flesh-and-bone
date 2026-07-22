"""CPU-fast render-configuration contracts for H5 orchestration."""

from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

import flesh_and_bone.h5_experiment as experiment


def test_render_seed_forwards_density_specific_radius_opacity_and_size(
    monkeypatch, tmp_path
):
    calls = []
    monkeypatch.setattr(experiment, "load_base_color", lambda _: object())
    monkeypatch.setattr(
        experiment,
        "sample_texture",
        lambda image, uv: np.zeros((uv.shape[0], 3), dtype=np.float32),
    )

    def fake_render(*args, **kwargs):
        calls.append(kwargs)
        return Image.new("RGB", (2, 2))

    monkeypatch.setattr(experiment, "render_colored_splats", fake_render)
    monkeypatch.setattr(experiment, "save_gif", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        experiment, "save_contact_sheet", lambda *args, **kwargs: None
    )
    asset = SimpleNamespace(
        bone_endpoints=torch.zeros(29, 1, 2, 3)
    )
    volume = SimpleNamespace(
        uv=torch.zeros(1, 2),
        splat_scale=torch.ones(1),
        metadata={"pitch": 0.0125},
    )
    trajectory = SimpleNamespace(
        residual=torch.zeros(29, 1, 1, 3),
        lbs_positions=torch.zeros(29, 1, 3),
    )
    rollout = torch.zeros(3, 29, 1, 3)
    experiment._render_seed(
        tmp_path,
        asset,
        volume,
        trajectory,
        rollout,
        7,
        "unused.zip",
        720,
        0.50,
        0.72,
    )
    assert len(calls) == 68
    assert all(call["size"] == 720 for call in calls)
    assert all(call["opacity"] == 0.72 for call in calls)
    assert all(
        call["splat_radius"] == 0.50 * 0.0125 for call in calls
    )


def test_run_h5_passes_teacher_config_to_both_rollout_arms(
    monkeypatch, tmp_path
):
    teacher_config = experiment.ElasticTeacherConfig()
    training_config = experiment.FleshTrainingConfig(steps=1)
    volume = SimpleNamespace(
        cell_count=1,
        points=torch.zeros(1, 3),
        metadata={"pitch": 0.025},
    )
    trajectory = object()
    graph = object()
    rollout_configs = []

    monkeypatch.setattr(experiment, "_device", lambda _: torch.device("cpu"))
    monkeypatch.setattr(experiment, "load_rig_asset", lambda *a, **k: object())
    monkeypatch.setattr(experiment, "load_h4_volume", lambda *a, **k: volume)
    monkeypatch.setattr(experiment, "build_voxel_graph", lambda *a, **k: graph)
    monkeypatch.setattr(
        experiment, "simulate_teacher", lambda *a, **k: trajectory
    )
    monkeypatch.setattr(
        experiment,
        "measure_teacher",
        lambda *a, **k: {
            "graph_directed_edges": 1,
            "residual_rms": 0.01,
            "far_near_amplitude_ratio": 2.0,
            "cycle_seam_rms": 0.0,
        },
    )
    rule = SimpleNamespace(state_dict=lambda: {})
    monkeypatch.setattr(
        experiment,
        "train_flesh_rule",
        lambda *a, **k: (rule, {"holdout_acceleration_nrmse": 0.0}),
    )

    def fake_rollout(rule, trajectory, volume, graph, config, **kwargs):
        rollout_configs.append(config)
        return torch.zeros(1, 1, 1, 3), None, None

    monkeypatch.setattr(experiment, "rollout_flesh_rule", fake_rollout)
    monkeypatch.setattr(
        experiment, "measure_rollout", lambda *a, **k: {"position_rms": 0.0}
    )
    monkeypatch.setattr(
        experiment, "acceptance_h5", lambda *a, **k: {"pass": True}
    )
    config = experiment.H5Config(seeds=(5,), render_seed=7)
    report = experiment.run_h5(
        tmp_path,
        config,
        teacher_config=teacher_config,
        training_config=training_config,
    )
    assert report["acceptance"]["pass"]
    assert rollout_configs == [teacher_config, teacher_config]
