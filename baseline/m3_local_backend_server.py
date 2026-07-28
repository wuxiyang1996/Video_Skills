#!/usr/bin/env python3
"""Serve local M3 embedding and Whisper models to isolated M3 environments."""

from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .m3_local_backends import encode_texts, get_whisper_with_retry, validate_local_models


class Handler(BaseHTTPRequestHandler):
    def _write(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._write(200, {"ok": True, "models": validate_local_models()})
        else:
            self._write(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if self.path == "/embed":
                self._write(200, {"embeddings": encode_texts(list(payload["texts"]))})
            elif self.path == "/transcribe":
                text = get_whisper_with_retry("local", str(payload["file_path"]))
                self._write(200, {"text": text})
            else:
                self._write(404, {"error": "not found"})
        except Exception as exc:
            self._write(500, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} {format % args}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18765)
    args = parser.parse_args()
    os.environ.pop("M3_LOCAL_BACKEND_URL", None)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(json.dumps({"listening": f"http://{args.host}:{args.port}"}), flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
