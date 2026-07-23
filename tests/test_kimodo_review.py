from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

import numpy as np
import pytest

from flesh_and_bone.kimodo_review import (
    KimodoGenerationClient,
    KimodoReviewConfig,
    ReviewJobManager,
    validate_review_request,
)


class _KimodoStub(BaseHTTPRequestHandler):
    last_payload = None

    def log_message(self, format, *args):
        return

    def do_GET(self):
        body = json.dumps(
            {"status": "ok", "model": "stub-model", "version": "test"}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        type(self).last_payload = json.loads(self.rfile.read(length))
        from io import BytesIO

        buffer = BytesIO()
        np.savez(buffer, local_rot_mats=np.eye(3)[None, None])
        body = buffer.getvalue()
        self.send_response(200)
        self.send_header("Content-Type", "application/x-kimodo-motion+npz")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def kimodo_stub():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _KimodoStub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_client_sends_generation_controls_and_returns_npz(kimodo_stub):
    client = KimodoGenerationClient(kimodo_stub, timeout=2)

    assert client.health()["model"] == "stub-model"
    body = client.generate("walk", 4.5, 17, 72, postprocess=False)

    assert body.startswith(b"PK")
    assert _KimodoStub.last_payload == {
        "prompt": "walk",
        "duration_s": 4.5,
        "seed": 17,
        "diffusion_steps": 72,
        "postprocess": False,
    }


def test_request_validation_bounds_and_random_seed():
    values = validate_review_request({"prompt": "  wave  ", "seed": ""})

    assert values["prompt"] == "wave"
    assert 0 <= values["seed"] < 2**31
    assert values["apply_contact_ik"] is True
    with pytest.raises(ValueError, match="duration_s"):
        validate_review_request({"prompt": "wave", "duration_s": 31})
    with pytest.raises(ValueError, match="diffusion_steps"):
        validate_review_request({"prompt": "wave", "diffusion_steps": 0})


def test_completed_manifests_reload_and_decisions_persist(tmp_path):
    output = tmp_path / "review"
    job_dir = output / "existing-job"
    job_dir.mkdir(parents=True)
    manifest = {
        "created_at": "2026-01-01T00:00:00+00:00",
        "request": {"prompt": "walk", "seed": 1},
        "decision": {"status": "unreviewed", "note": "", "updated_at": None},
    }
    (job_dir / "manifest.json").write_text(json.dumps(manifest))
    config = KimodoReviewConfig(output_root=output)
    manager = ReviewJobManager(Path.cwd(), config=config)

    job = manager.decide("existing-job", "rejected", "left leg crosses torso")

    assert job["result"]["decision"]["status"] == "rejected"
    persisted = json.loads((job_dir / "manifest.json").read_text())
    assert persisted["decision"]["note"] == "left leg crosses torso"
