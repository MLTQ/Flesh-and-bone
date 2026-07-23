"""Generate, retarget, diagnose, and preview Kimodo motion review jobs."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
import hashlib
import json
from pathlib import Path
import queue
import sys
import threading
import traceback
import uuid

import numpy as np
import torch

from .h4_volume import load_h4_volume
from .kimodo_client import (
    KimodoGenerationClient,
    KimodoReviewConfig,
    validate_review_request,
)
from .kimodo_diagnostics import analyze_retarget
from .kimodo_preview import render_review_preview
from .retarget_skin import canonical_motion_skin
from .rig_asset import load_rig_asset


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _motion_api(archepelago_root: Path):
    motion_root = archepelago_root / "backend" / "motion"
    if not motion_root.exists():
        raise FileNotFoundError(f"Archepelago motion package not found: {motion_root}")
    if str(motion_root) not in sys.path:
        sys.path.insert(0, str(motion_root))
    from skeletor_motion.retarget import (
        _fk_world,
        apply_foot_contacts,
        build_retarget_map,
        kimodo_npz_to_animation,
        load_soma_rig,
        retarget_animation,
    )
    from skeletor_motion.skeleton import Rig
    from skeletor_motion.skeleton_profile import profile_rig
    from skeletor_motion.transforms import rotation_6d_to_matrix_np

    return {
        "_fk_world": _fk_world,
        "apply_foot_contacts": apply_foot_contacts,
        "build_retarget_map": build_retarget_map,
        "kimodo_npz_to_animation": kimodo_npz_to_animation,
        "load_soma_rig": load_soma_rig,
        "retarget_animation": retarget_animation,
        "Rig": Rig,
        "profile_rig": profile_rig,
        "rotation_6d_to_matrix_np": rotation_6d_to_matrix_np,
        "motion_root": motion_root,
    }


def _destination_rig(asset, Rig):
    heads = asset.rest_bone_endpoints[:, 0].detach().cpu().numpy()
    parents = asset.bone_parents.detach().cpu().numpy()
    offsets = np.empty_like(heads, dtype=np.float32)
    for joint, parent in enumerate(parents):
        offsets[joint] = heads[joint] if parent < 0 else heads[joint] - heads[parent]
    return Rig(
        parents=parents,
        rest_offsets=offsets,
        joint_names=list(asset.bone_names),
    )


def _raw_fk_error(raw_data: dict, source_rig, fk_world) -> float | None:
    required = {"local_rot_mats", "root_positions", "posed_joints"}
    if not required.issubset(raw_data):
        return None
    _, reconstructed = fk_world(
        source_rig.parents,
        source_rig.rest_offsets,
        np.asarray(raw_data["local_rot_mats"], dtype=np.float32),
        np.asarray(raw_data["root_positions"], dtype=np.float32),
    )
    reference = np.asarray(raw_data["posed_joints"], dtype=np.float32)
    if reference.shape != reconstructed.shape:
        return None
    return float(np.linalg.norm(reconstructed - reference, axis=-1).max())


def run_review_job(
    request_values: dict,
    job_dir: Path,
    config: KimodoReviewConfig,
    client: KimodoGenerationClient,
    update=lambda stage, progress, message: None,
) -> dict:
    """Execute one reproducible generation-to-preview review pipeline."""
    values = validate_review_request(request_values)
    job_dir.mkdir(parents=True, exist_ok=True)
    update("checking_server", 0.04, "Checking Kimodo and local assets")
    server_health = client.health()
    api = _motion_api(config.archepelago_root)
    for path in (config.rig_path, config.volume_path, config.texture_archive):
        if not path.exists():
            raise FileNotFoundError(path)

    update("generating", 0.10, "Kimodo is diffusing the motion sequence")
    raw_bytes = client.generate(
        values["prompt"],
        values["duration_s"],
        values["seed"],
        values["diffusion_steps"],
        values["postprocess"],
    )
    raw_path = job_dir / "raw_kimodo.npz"
    raw_path.write_bytes(raw_bytes)
    with np.load(BytesIO(raw_bytes), allow_pickle=False) as bundle:
        raw_data = {name: np.array(bundle[name]) for name in bundle.files}

    update("retargeting", 0.50, "Mapping SOMA roles onto the Meshy rig")
    asset = load_rig_asset(config.rig_path, dtype=torch.float32)
    destination = _destination_rig(asset, api["Rig"])
    source = api["load_soma_rig"](
        api["motion_root"] / "reports" / "kimodo" / "soma_skeleton.json"
    )
    mapping = api["build_retarget_map"](source, destination)
    raw_animation = api["kimodo_npz_to_animation"](raw_data)
    retargeted = api["retarget_animation"](mapping, raw_animation)
    contacts = raw_data.get("foot_contacts")
    contact_ik_applied = bool(values["apply_contact_ik"] and contacts is not None)
    if contact_ik_applied:
        update("contact_ik", 0.60, "Locking source-declared planted feet")
        retargeted = api["apply_foot_contacts"](
            destination, retargeted, contacts
        )
    local_np = api["rotation_6d_to_matrix_np"](retargeted.rotations_6d)
    local = torch.as_tensor(local_np, dtype=torch.float32)
    root = torch.as_tensor(retargeted.root_positions, dtype=torch.float32)
    skin, endpoint_tensor = canonical_motion_skin(asset, local, root)
    endpoints = endpoint_tensor.detach().cpu().numpy()

    update("diagnostics", 0.68, "Measuring retarget anatomy and invariants")
    diagnostics = analyze_retarget(
        endpoints,
        asset.bone_names,
        asset.bone_parents.detach().cpu().numpy(),
        contacts=contacts if contact_ik_applied else None,
    )
    raw_fk_error = _raw_fk_error(raw_data, source, api["_fk_world"])
    if raw_fk_error is not None:
        diagnostics["context"]["raw_fk_reconstruction_max_m"] = raw_fk_error
        if raw_fk_error > 0.0005:
            diagnostics["verdict"] = "fail"
            diagnostics["reasons"].append("Raw FK reconstruction")
            diagnostics["metrics"].append(
                {
                    "id": "raw_fk_reconstruction",
                    "label": "Raw FK reconstruction",
                    "value": raw_fk_error,
                    "unit": "m",
                    "status": "fail",
                    "detail": "SOMA local rotations do not reconstruct posed_joints.",
                }
            )

    mapped_pairs = {
        destination.joint_names[dst]: source.joint_names[src]
        for dst, src in mapping.dst_to_src.items()
    }
    motion_metadata = {
        "format": "flesh-and-bone-kimodo-review-motion-v1",
        "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "prompt": values["prompt"],
        "seed": values["seed"],
        "fps": int(retargeted.fps),
        "frame_count": int(local.shape[0]),
        "mapped_roles": list(mapping.mapped_roles),
        "mapped_role_count": len(mapping.mapped_roles),
        "root_scale": float(mapping.root_scale),
        "contact_ik_applied": contact_ik_applied,
    }
    save_values = {
        "metadata_json": np.asarray(json.dumps(motion_metadata)),
        "local_rotations": local_np,
        "root_positions": np.asarray(retargeted.root_positions),
        "bone_endpoints": endpoints,
    }
    if contacts is not None:
        save_values["foot_contacts"] = np.asarray(contacts)
    np.savez_compressed(job_dir / "retargeted_motion.npz", **save_values)

    update("skinning", 0.75, "Skinning a bounded dense-cell preview")
    volume = load_h4_volume(config.volume_path, dtype=torch.float32)
    update("rendering", 0.82, "Rendering the animation and diagnostic frames")
    render_info = render_review_preview(
        job_dir,
        config,
        asset,
        volume,
        skin,
        endpoints,
        np.asarray(retargeted.root_positions),
        int(retargeted.fps),
        diagnostics["worst_frame"],
    )
    manifest = {
        "format": "flesh-and-bone-kimodo-review-v1",
        "created_at": _utc_now(),
        "request": values,
        "server": server_health,
        "motion": motion_metadata,
        "retarget": {
            "destination_profile": api["profile_rig"](destination).to_dict(),
            "mapped_pairs": mapped_pairs,
        },
        "diagnostics": diagnostics,
        "render": render_info,
        "artifacts": {
            "animation": "character.gif",
            "contact_sheet": "contact_sheet.png",
            "anatomy_frame": "anatomy_frame.png",
            "raw_motion": "raw_kimodo.npz",
            "retargeted_motion": "retargeted_motion.npz",
        },
        "decision": {"status": "unreviewed", "note": "", "updated_at": None},
    }
    (job_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


class ReviewJobManager:
    """Single-worker job queue with durable manifests and pollable state."""

    def __init__(
        self,
        repository_root: Path,
        config: KimodoReviewConfig | None = None,
        client: KimodoGenerationClient | None = None,
    ):
        self.repository_root = Path(repository_root).resolve()
        self.config = (config or KimodoReviewConfig()).resolved(self.repository_root)
        self.client = client or KimodoGenerationClient(self.config.server_url)
        self.config.output_root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue[str] = queue.Queue()
        self._load_completed_jobs()
        self._worker = threading.Thread(target=self._work, daemon=True)
        self._worker.start()

    def _load_completed_jobs(self):
        for manifest_path in sorted(
            self.config.output_root.glob("*/manifest.json"), reverse=True
        ):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            job_id = manifest_path.parent.name
            self._jobs[job_id] = {
                "id": job_id,
                "status": "complete",
                "stage": "complete",
                "progress": 1.0,
                "message": "Review artifacts ready",
                "created_at": manifest.get("created_at", _utc_now()),
                "updated_at": manifest.get("created_at", _utc_now()),
                "request": manifest.get("request", {}),
                "result": manifest,
                "error": None,
            }

    def submit(self, request_values: dict) -> dict:
        values = validate_review_request(request_values)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        job_id = f"{timestamp}-{uuid.uuid4().hex[:8]}"
        now = _utc_now()
        job = {
            "id": job_id,
            "status": "queued",
            "stage": "queued",
            "progress": 0.0,
            "message": "Waiting for the single Kimodo generation slot",
            "created_at": now,
            "updated_at": now,
            "request": values,
            "result": None,
            "error": None,
        }
        with self._lock:
            self._jobs[job_id] = job
        self._queue.put(job_id)
        return self.get(job_id)

    def get(self, job_id: str) -> dict:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return json.loads(json.dumps(self._jobs[job_id]))

    def list(self) -> list[dict]:
        with self._lock:
            values = sorted(
                self._jobs.values(), key=lambda item: item["created_at"], reverse=True
            )
            return json.loads(json.dumps(values))

    def _update(self, job_id: str, stage: str, progress: float, message: str):
        with self._lock:
            job = self._jobs[job_id]
            job.update(
                {
                    "status": "running",
                    "stage": stage,
                    "progress": float(progress),
                    "message": message,
                    "updated_at": _utc_now(),
                }
            )

    def _work(self):
        while True:
            job_id = self._queue.get()
            try:
                job = self.get(job_id)
                result = run_review_job(
                    job["request"],
                    self.config.output_root / job_id,
                    self.config,
                    self.client,
                    lambda stage, progress, message: self._update(
                        job_id, stage, progress, message
                    ),
                )
                with self._lock:
                    self._jobs[job_id].update(
                        {
                            "status": "complete",
                            "stage": "complete",
                            "progress": 1.0,
                            "message": "Review artifacts ready",
                            "updated_at": _utc_now(),
                            "result": result,
                        }
                    )
            except Exception as exc:  # surfaced to the UI with a saved trace
                job_dir = self.config.output_root / job_id
                job_dir.mkdir(parents=True, exist_ok=True)
                trace = traceback.format_exc()
                (job_dir / "error.txt").write_text(trace, encoding="utf-8")
                with self._lock:
                    self._jobs[job_id].update(
                        {
                            "status": "failed",
                            "stage": "failed",
                            "message": "Generation or retargeting failed",
                            "updated_at": _utc_now(),
                            "error": str(exc),
                        }
                    )
            finally:
                self._queue.task_done()

    def decide(self, job_id: str, status: str, note: str) -> dict:
        if status not in {"unreviewed", "accepted", "rejected"}:
            raise ValueError("decision status must be unreviewed, accepted, or rejected")
        if len(note) > 2000:
            raise ValueError("decision note must be at most 2000 characters")
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job["status"] != "complete" or job["result"] is None:
                raise ValueError("only complete jobs can be reviewed")
            decision = {
                "status": status,
                "note": note.strip(),
                "updated_at": _utc_now(),
            }
            job["result"]["decision"] = decision
            job["updated_at"] = decision["updated_at"]
            manifest = job["result"]
            (self.config.output_root / job_id / "manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
        return self.get(job_id)
