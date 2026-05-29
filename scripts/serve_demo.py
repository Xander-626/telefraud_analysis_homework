"""Serve the TeleAntiFraud local frontend demo."""

from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
WEB_ROOT = ROOT / "web" / "demo"

sys.path.insert(0, str(SRC))

from teledeceit.demo_backend import build_text_prediction, list_demo_samples, predict_demo_sample


def build_api_response(method: str, path: str, body: bytes) -> tuple[int, dict[str, Any]]:
    """Build a JSON response for demo API routes."""

    if method == "GET" and path == "/api/demo/samples":
        return 200, {"samples": list_demo_samples()}

    if method == "POST" and path == "/api/demo/predict-sample":
        payload = _json_body(body)
        sample_id = str(payload.get("sample_id", ""))
        try:
            return 200, predict_demo_sample(sample_id)
        except KeyError:
            return 404, {"error": f"Unknown sample_id: {sample_id}"}

    if method == "POST" and path == "/api/predict":
        payload = _json_body(body)
        text = payload.get("text")
        if isinstance(text, str):
            try:
                return 200, build_text_prediction(text)
            except ValueError as exc:
                return 400, {"error": str(exc)}
        return 501, {"error": "Audio upload is a secondary demo path and is not enabled in this lightweight server."}

    return 404, {"error": f"No demo API route for {method} {path}"}


class DemoRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self._send_json(*build_api_response("GET", parsed.path, b""))
            return
        self._send_static(parsed.path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        self._send_json(*build_api_response("POST", parsed.path, body))

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_common_headers()
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("[demo] " + format % args + "\n")

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._send_common_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_static(self, request_path: str) -> None:
        relative = "index.html" if request_path in {"", "/"} else request_path.lstrip("/")
        target = (WEB_ROOT / relative).resolve()
        if WEB_ROOT.resolve() not in target.parents and target != WEB_ROOT.resolve():
            self.send_error(403)
            return
        if not target.exists() or not target.is_file():
            self.send_error(404)
            return

        content = target.read_bytes()
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        if target.suffix == ".js":
            content_type = "text/javascript"
        self.send_response(200)
        self._send_common_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_common_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), DemoRequestHandler)
    print(f"Serving TeleAntiFraud demo at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping demo server")


def _json_body(body: bytes) -> dict[str, Any]:
    if not body:
        return {}
    try:
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


if __name__ == "__main__":
    main()
