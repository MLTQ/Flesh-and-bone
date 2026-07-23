#!/usr/bin/env python3
"""Serve the local Kimodo generation and retarget review console."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
from urllib.parse import unquote, urlparse

from flesh_and_bone.kimodo_review import KimodoReviewConfig, ReviewJobManager


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = Path(__file__).resolve().parent / "kimodo_review_ui"


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--kimodo-url", default="http://192.168.0.202:8111")
    parser.add_argument(
        "--archepelago",
        type=Path,
        default=Path.home() / "Code" / "Archepelago",
    )
    parser.add_argument("--preview-size", type=int, default=384)
    parser.add_argument("--preview-frames", type=int, default=45)
    parser.add_argument("--preview-cells", type=int, default=24000)
    return parser.parse_args()


class ReviewHandler(BaseHTTPRequestHandler):
    manager: ReviewJobManager

    def log_message(self, format, *args):
        print(f"[kimodo-review] {self.address_string()} {format % args}")

    def _json(self, value, status=HTTPStatus.OK):
        payload = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _error(self, status, message):
        self._json({"error": str(message)}, status)

    def _body(self):
        length = int(self.headers.get("Content-Length", "0"))
        if length > 1_000_000:
            raise ValueError("request body is too large")
        data = self.rfile.read(length)
        return json.loads(data or b"{}")

    def _file(self, path: Path, cache=True):
        if not path.is_file():
            self._error(HTTPStatus.NOT_FOUND, "file not found")
            return
        payload = path.read_bytes()
        mime, _ = mimetypes.guess_type(path.name)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime or "application/octet-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header(
            "Cache-Control", "public, max-age=3600" if cache else "no-store"
        )
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        try:
            if path == "/api/health":
                try:
                    remote = self.manager.client.health()
                    self._json({"local": "ok", "kimodo": remote})
                except Exception as exc:
                    self._json(
                        {"local": "ok", "kimodo": {"status": "unavailable", "error": str(exc)}},
                        HTTPStatus.SERVICE_UNAVAILABLE,
                    )
                return
            if path == "/api/jobs":
                self._json({"jobs": self.manager.list()})
                return
            if path.startswith("/api/jobs/"):
                job_id = path.removeprefix("/api/jobs/").strip("/")
                self._json(self.manager.get(job_id))
                return
            if path.startswith("/artifacts/"):
                relative = Path(path.removeprefix("/artifacts/"))
                candidate = (self.manager.config.output_root / relative).resolve()
                root = self.manager.config.output_root.resolve()
                if root not in candidate.parents:
                    self._error(HTTPStatus.BAD_REQUEST, "invalid artifact path")
                    return
                self._file(candidate, cache=False)
                return
            static_name = "index.html" if path in {"/", "/index.html"} else path.lstrip("/")
            candidate = (STATIC_ROOT / static_name).resolve()
            if STATIC_ROOT.resolve() not in candidate.parents:
                self._error(HTTPStatus.BAD_REQUEST, "invalid static path")
                return
            self._file(candidate)
        except KeyError:
            self._error(HTTPStatus.NOT_FOUND, "job not found")
        except Exception as exc:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, exc)

    def do_POST(self):
        path = unquote(urlparse(self.path).path)
        try:
            values = self._body()
            if path == "/api/jobs":
                self._json(self.manager.submit(values), HTTPStatus.ACCEPTED)
                return
            if path.startswith("/api/jobs/") and path.endswith("/decision"):
                job_id = path.removeprefix("/api/jobs/").removesuffix("/decision").strip("/")
                job = self.manager.decide(
                    job_id, str(values.get("status", "unreviewed")), str(values.get("note", ""))
                )
                self._json(job)
                return
            self._error(HTTPStatus.NOT_FOUND, "route not found")
        except KeyError:
            self._error(HTTPStatus.NOT_FOUND, "job not found")
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, exc)
        except Exception as exc:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, exc)


def main():
    args = _arguments()
    config = KimodoReviewConfig(
        server_url=args.kimodo_url,
        archepelago_root=args.archepelago,
        preview_size=args.preview_size,
        preview_frames=args.preview_frames,
        preview_cells=args.preview_cells,
    )
    ReviewHandler.manager = ReviewJobManager(REPOSITORY_ROOT, config=config)
    server = ThreadingHTTPServer((args.host, args.port), ReviewHandler)
    print(f"Kimodo review console: http://{args.host}:{args.port}")
    print(f"Kimodo generation API: {args.kimodo_url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Kimodo review console")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
