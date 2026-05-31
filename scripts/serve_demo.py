"""Serve the TeleAntiFraud local frontend demo."""

from __future__ import annotations

import argparse
import json
import mimetypes
import subprocess
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
WEB_ROOT = ROOT / "web" / "demo"

sys.path.insert(0, str(SRC))

from teledeceit.demo_backend import (
    build_text_prediction,
    build_upload_prediction,
    get_local_audio_path,
    list_demo_samples,
    predict_demo_sample,
)


def build_api_response(
    method: str, path: str, body: bytes, headers: dict[str, str] | None = None
) -> tuple[int, dict[str, Any]]:
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

    if method == "POST" and path == "/api/predict/upload":
        return _handle_upload(body, headers or {})

    if method == "POST" and path == "/api/predict":
        payload = _json_body(body)
        text = payload.get("text")
        if isinstance(text, str):
            try:
                return 200, build_text_prediction(text)
            except ValueError as exc:
                return 400, {"error": str(exc)}
        return 400, {"error": "text field is required"}

    return 404, {"error": f"No demo API route for {method} {path}"}


def _handle_upload(body: bytes, headers: dict[str, str]) -> tuple[int, dict[str, Any]]:
    """Parse multipart upload and run demo prediction."""
    content_type = headers.get("Content-Type", headers.get("content-type", ""))
    if "multipart/form-data" not in content_type:
        # Try raw binary upload (the file content directly)
        return _predict_uploaded_audio(body, "uploaded_audio")

    # Extract boundary from Content-Type: multipart/form-data; boundary=----XXX
    boundary = None
    for part in content_type.split(";"):
        part = part.strip()
        if part.startswith("boundary="):
            boundary = part[len("boundary="):].strip('"')
            break

    if not boundary:
        return 400, {"error": "Could not parse multipart boundary"}

    boundary_bytes = boundary.encode("utf-8")
    parts = body.split(b"--" + boundary_bytes)
    for part in parts:
        if b"Content-Disposition" not in part:
            continue
        # Find the double CRLF that separates headers from body
        header_end = part.find(b"\r\n\r\n")
        if header_end == -1:
            continue
        file_data = part[header_end + 4:]
        # Strip trailing boundary markers
        if file_data.endswith(b"\r\n"):
            file_data = file_data[:-2]
        if file_data.endswith(b"--"):
            file_data = file_data[:-2]
        if file_data.endswith(b"\r\n"):
            file_data = file_data[:-2]

        if len(file_data) > 44:  # minimum MP3 header size
            # Extract filename from Content-Disposition
            headers_section = part[:header_end].decode("utf-8", errors="replace")
            filename = "uploaded_audio"
            for line in headers_section.split("\r\n"):
                if "filename=" in line:
                    fname_start = line.find('filename="') + 10
                    fname_end = line.find('"', fname_start)
                    if fname_start >= 10 and fname_end > fname_start:
                        filename = line[fname_start:fname_end]
                    break
            return _predict_uploaded_audio(file_data, filename)

    return 400, {"error": "No file data found in upload"}


def _predict_uploaded_audio(file_data: bytes, filename: str) -> tuple[int, dict[str, Any]]:
    """Run demo prediction on an uploaded audio file."""
    # Write to a temporary file for ffprobe analysis
    suffix = Path(filename).suffix or ".mp3"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_data)
        tmp_path = tmp.name

    try:
        duration = None
        try:
            out = subprocess.check_output(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", tmp_path],
                stderr=subprocess.DEVNULL, timeout=10,
            )
            duration = float(out.decode().strip())
        except Exception:
            pass

        result = build_upload_prediction(
            filename=filename,
            file_size=len(file_data),
            duration=duration,
        )
        return 200, result
    finally:
        Path(tmp_path).unlink(missing_ok=True)


class DemoRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        # Audio streaming endpoint — serve local MP3 files
        if parsed.path.startswith("/api/audio/"):
            sample_id = parsed.path.split("/api/audio/")[-1]
            self._send_audio(sample_id)
            return
        if parsed.path.startswith("/api/"):
            self._send_json(*build_api_response("GET", parsed.path, b""))
            return
        self._send_static(parsed.path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        # Pass headers dict for multipart boundary parsing
        req_headers = {k: v for k, v in self.headers.items()}
        self._send_json(*build_api_response("POST", parsed.path, body, req_headers))

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

    def _send_audio(self, sample_id: str) -> None:
        local_path = get_local_audio_path(sample_id)
        if local_path is None:
            self.send_error(404, f"No local audio for sample {sample_id}")
            return
        audio_file = ROOT / "data" / local_path
        if not audio_file.exists():
            self.send_error(404, f"Audio file not found: {local_path}")
            return
        try:
            content = audio_file.read_bytes()
            self.send_response(200)
            self._send_common_headers()
            self.send_header("Content-Type", "audio/mpeg")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(content)
        except OSError:
            self.send_error(500, "Failed to read audio file")


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
