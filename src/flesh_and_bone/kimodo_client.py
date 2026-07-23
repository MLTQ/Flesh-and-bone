"""Configuration, validation, and HTTP access for Kimodo review jobs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import secrets
from urllib import error, request as urlrequest


@dataclass(frozen=True)
class KimodoReviewConfig:
    """Paths and preview bounds for the local review service."""

    server_url: str = "http://192.168.0.202:8111"
    archepelago_root: Path = Path.home() / "Code" / "Archepelago"
    rig_path: Path = Path("model/derived/meshy_blonde_h4_rig.npz")
    volume_path: Path = Path(
        "model/derived/meshy_blonde_h4_volume_p0125.npz"
    )
    texture_archive: Path = Path(
        "model/Meshy_AI_Blonde_female_mechani_biped.zip"
    )
    output_root: Path = Path("experiments/runs/kimodo_review")
    preview_size: int = 384
    preview_frames: int = 45
    preview_cells: int = 24000
    render_splat_radius_scale: float = 0.50

    def resolved(self, repository_root: Path) -> "KimodoReviewConfig":
        data = asdict(self)
        for key in ("rig_path", "volume_path", "texture_archive", "output_root"):
            value = Path(data[key])
            data[key] = value if value.is_absolute() else repository_root / value
        data["archepelago_root"] = Path(data["archepelago_root"])
        return KimodoReviewConfig(**data)


class KimodoGenerationClient:
    """Minimal dependency-free client for the synchronous Kimodo HTTP API."""

    def __init__(self, server_url: str, timeout: float = 180.0):
        self.server_url = server_url.rstrip("/")
        self.timeout = float(timeout)

    def _request(self, endpoint: str, data: bytes | None = None):
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        outgoing = urlrequest.Request(
            f"{self.server_url}{endpoint}", data=data, headers=headers
        )
        try:
            with urlrequest.urlopen(outgoing, timeout=self.timeout) as response:
                return response.read(), response.headers.get_content_type()
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Kimodo HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Kimodo server unavailable: {exc.reason}") from exc

    def health(self) -> dict:
        body, _ = self._request("/healthz")
        return json.loads(body)

    def generate(
        self,
        prompt: str,
        duration_s: float,
        seed: int,
        diffusion_steps: int,
        postprocess: bool = True,
    ) -> bytes:
        payload = json.dumps(
            {
                "prompt": prompt,
                "duration_s": float(duration_s),
                "seed": int(seed),
                "diffusion_steps": int(diffusion_steps),
                "postprocess": bool(postprocess),
            }
        ).encode("utf-8")
        body, content_type = self._request("/generate", payload)
        if "npz" not in content_type and not body.startswith(b"PK"):
            raise RuntimeError(
                f"Kimodo returned unexpected content type {content_type}"
            )
        return body


def validate_review_request(values: dict) -> dict:
    """Normalize the public job request and reject unsafe/extreme values."""
    prompt = str(values.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("prompt is required")
    if len(prompt) > 600:
        raise ValueError("prompt must be at most 600 characters")
    duration_s = float(values.get("duration_s", 6.0))
    if not 0.5 <= duration_s <= 30.0:
        raise ValueError("duration_s must be between 0.5 and 30")
    diffusion_steps = int(values.get("diffusion_steps", 50))
    if not 1 <= diffusion_steps <= 200:
        raise ValueError("diffusion_steps must be between 1 and 200")
    seed_value = values.get("seed")
    seed = secrets.randbelow(2**31) if seed_value in (None, "") else int(seed_value)
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be between 0 and 4294967295")
    return {
        "prompt": prompt,
        "duration_s": duration_s,
        "seed": seed,
        "diffusion_steps": diffusion_steps,
        "postprocess": bool(values.get("postprocess", True)),
        "apply_contact_ik": bool(values.get("apply_contact_ik", True)),
    }
