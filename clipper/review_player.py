"""Generic HTTP review player with byte-range (206) support and external insights."""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")
CHUNK_SIZE = 1024 * 1024
DEFAULT_INSIGHTS_DIR = Path(__file__).resolve().parent.parent / "insights"
TEMPLATE_PATH = Path(__file__).resolve().with_name("review_player.html")


def _load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _extract_clips_list(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        for key in ("clips", "highlights", "candidates", "items"):
            val = raw.get(key)
            if isinstance(val, list):
                return [item for item in val if isinstance(item, dict)]
    return []


def load_workdir_payload(workdir: Path) -> dict[str, Any]:
    """Load video metadata and candidate/highlight clips from a Clipper workdir."""
    info = _load_json(workdir / "source.info.json") or {}
    video_title = str(info.get("title") or workdir.name)
    video_url = str(info.get("webpage_url") or info.get("original_url") or "")
    yt_id = str(info.get("id") or "")

    raw_clips: list[dict[str, Any]] = []
    for candidate_name in ("review-clips.json", "highlights.json", "clips.json"):
        data = _load_json(workdir / candidate_name)
        extracted = _extract_clips_list(data)
        if extracted:
            raw_clips = extracted
            break
    if not raw_clips:
        for cand_file in sorted(workdir.glob("*.candidates.json")):
            extracted = _extract_clips_list(_load_json(cand_file))
            if extracted:
                raw_clips = extracted
                break

    clips: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_clips, start=1):
        start = float(item.get("start", 0.0))
        end = float(item.get("end", start))
        score = item.get("score")
        if score is None:
            score = item.get("selection_score", 0)
        webcam_segments = item.get("webcamSegments")
        if not webcam_segments and isinstance(item.get("webcam_crop"), list):
            crop = item["webcam_crop"]
            webcam_segments = [{"from": 0, "crop": crop, "label": f"Webcam ({','.join(map(str, crop))})"}]
        clips.append(
            {
                "num": int(item.get("num", idx)),
                "category": str(item.get("category") or ("focus" if item.get("focus") else "clip")),
                "start": start,
                "end": end,
                "score": round(float(score), 1) if isinstance(score, (int, float)) else 0,
                "title": str(item.get("title") or f"Corte #{idx}"),
                "hook": str(item.get("hook") or ""),
                "reason": str(item.get("reason") or item.get("description") or ""),
                "webcamSegments": webcam_segments or [],
                "webcamNote": str(item.get("webcamNote") or ""),
            }
        )

    return {
        "workdir_name": workdir.name,
        "video_title": video_title,
        "video_url": video_url,
        "yt_id": yt_id,
        "clips": clips,
    }


def make_handler(workdir: Path, insights_dir: Path):
    feedback_file = workdir / "review-feedback.json"
    insights_file = insights_dir / f"{workdir.name}-insights.md"

    class ReviewPlayerHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(workdir), **kwargs)

        def end_headers(self):
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/clips-data":
                self._send_json(HTTPStatus.OK, load_workdir_payload(workdir))
                return
            if parsed.path == "/load-review":
                data = _load_json(feedback_file)
                self._send_json(HTTPStatus.OK, {"feedback": data if isinstance(data, dict) else {}})
                return
            if parsed.path in ("/", "/player.html"):
                if TEMPLATE_PATH.is_file():
                    body = TEMPLATE_PATH.read_bytes()
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
            super().do_GET()

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path != "/save-review":
                self.send_error(HTTPStatus.NOT_FOUND, "Endpoint not found")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.send_error(HTTPStatus.BAD_REQUEST, "Invalid Content-Length")
                return
            if length <= 0 or length > 2_000_000:
                self.send_error(HTTPStatus.BAD_REQUEST, "Invalid payload size")
                return
            try:
                raw = self.rfile.read(length).decode("utf-8")
                payload = json.loads(raw)
            except (UnicodeDecodeError, ValueError) as exc:
                self.send_error(HTTPStatus.BAD_REQUEST, f"Invalid JSON: {exc}")
                return
            markdown = str(payload.get("markdown", ""))
            feedback = payload.get("feedback", {})
            if not isinstance(feedback, dict):
                self.send_error(HTTPStatus.BAD_REQUEST, "feedback must be an object")
                return
            insights_dir.mkdir(parents=True, exist_ok=True)
            feedback_file.write_text(json.dumps(feedback, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            insights_file.write_text(markdown.rstrip() + "\n", encoding="utf-8")
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "insights_path": str(insights_file),
                    "feedback_path": str(feedback_file),
                },
            )

        def guess_type(self, path: str) -> str:
            ext = Path(path).suffix.lower()
            if ext == ".mp4":
                return "video/mp4"
            if ext == ".html":
                return "text/html; charset=utf-8"
            if ext == ".json":
                return "application/json; charset=utf-8"
            guessed, _ = mimetypes.guess_type(path)
            return guessed or "application/octet-stream"

        def send_head(self):
            path = Path(self.translate_path(self.path))
            if not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "File not found")
                return None
            size = path.stat().st_size
            ctype = self.guess_type(str(path))
            range_header = self.headers.get("Range")
            if not range_header:
                file_obj = path.open("rb")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(size))
                self.end_headers()
                return file_obj

            match = RANGE_RE.match(range_header.strip())
            if not match:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, "Invalid Range header")
                return None
            start_s, end_s = match.groups()
            if not start_s and not end_s:
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, "Empty Range header")
                return None
            if start_s:
                start = int(start_s)
                end = int(end_s) if end_s else size - 1
            else:
                suffix = int(end_s)
                start = max(0, size - suffix)
                end = size - 1
            if start >= size or end < start or start < 0:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return None
            end = min(end, size - 1)
            length = end - start + 1
            file_obj = path.open("rb")
            file_obj.seek(start)
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(length))
            self.end_headers()
            return file_obj

        def copyfile(self, source, outputfile):
            if self.headers.get("Range"):
                content_length = None
                for header in self._headers_buffer:
                    if header.lower().startswith(b"content-length:"):
                        content_length = int(header.split(b":", 1)[1].strip())
                        break
                if content_length is not None:
                    remaining = content_length
                    while remaining > 0:
                        chunk = source.read(min(CHUNK_SIZE, remaining))
                        if not chunk:
                            break
                        outputfile.write(chunk)
                        remaining -= len(chunk)
                    return
            super().copyfile(source, outputfile)

    return ReviewPlayerHandler


def serve_review_player(workdir: Path, port: int = 8765, insights_dir: Path | None = None) -> None:
    resolved_workdir = workdir.resolve()
    resolved_insights = (insights_dir or DEFAULT_INSIGHTS_DIR).resolve()
    handler_cls = make_handler(resolved_workdir, resolved_insights)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler_cls)
    print(f"Clipper Review Player rodando em http://127.0.0.1:{port}/")
    print(f"Workdir: {resolved_workdir}")
    print(f"Insights dir: {resolved_insights}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve generic Clipper review player.")
    parser.add_argument("workdir", type=Path, help="Clipper output workdir containing source.mp4 and highlights.json")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--insights-dir", type=Path, default=DEFAULT_INSIGHTS_DIR)
    args = parser.parse_args(argv)
    serve_review_player(args.workdir, port=args.port, insights_dir=args.insights_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
