"""Zero-dependency local web UI for curating highlights and webcam crops."""

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from urllib.parse import urlparse

from clipper.reframe import _save_cached_webcam

_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")

HTML_PAGE = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Clipper Review</title>
<style>
  :root { --bg:#0f1117; --card:#181b24; --border:#2a3040; --text:#eef1f7; --muted:#9aa3b8; --accent:#a748f5; --ok:#22c55e; --warn:#f59e0b; --err:#ef4444; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--text); display:grid; grid-template-columns:1.2fr 1fr; height:100vh; }
  header { grid-column:1 / -1; padding:12px 18px; border-bottom:1px solid var(--border); display:flex; align-items:center; justify-content:space-between; background:#141720; }
  header h1 { margin:0; font-size:16px; font-weight:600; }
  .toolbar { display:flex; gap:8px; align-items:center; }
  button { background:var(--accent); color:#fff; border:none; border-radius:6px; padding:7px 12px; font-weight:600; cursor:pointer; font-size:13px; }
  button.secondary { background:#272d3d; }
  button.danger { background:var(--err); }
  button:hover { filter:brightness(1.1); }
  .left { padding:16px; display:flex; flex-direction:column; gap:12px; overflow:auto; border-right:1px solid var(--border); }
  .video-wrap { position:relative; width:100%; background:#000; border-radius:8px; overflow:hidden; }
  video { width:100%; display:block; max-height:60vh; }
  canvas.crop-overlay { position:absolute; inset:0; width:100%; height:100%; cursor:crosshair; display:none; }
  canvas.crop-overlay.active { display:block; }
  .right { padding:16px; overflow:auto; display:flex; flex-direction:column; gap:10px; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:8px; padding:12px; display:flex; flex-direction:column; gap:8px; }
  .card.active { border-color:var(--accent); }
  .row { display:flex; gap:8px; align-items:center; }
  .row label { font-size:12px; color:var(--muted); display:flex; flex-direction:column; gap:3px; flex:1; }
  input, textarea { background:#0f1117; color:var(--text); border:1px solid var(--border); border-radius:5px; padding:6px 8px; font-size:13px; width:100%; }
  .status { font-size:12px; color:var(--muted); }
  .badge { font-size:11px; padding:2px 7px; border-radius:999px; background:#272d3d; color:#d7dcf0; }
</style>
</head>
<body>
<header>
  <h1 id="title">Clipper Review</h1>
  <div class="toolbar">
    <span class="status" id="status">Carregando...</span>
    <button class="secondary" id="toggleCrop">Desenhar Webcam Crop</button>
    <button class="secondary" id="addClip">+ Novo corte</button>
    <button id="saveBtn">Salvar highlights.json</button>
  </div>
</header>
<section class="left">
  <div class="video-wrap">
    <video id="player" controls src="/video"></video>
    <canvas id="cropCanvas" class="crop-overlay"></canvas>
  </div>
  <div class="row">
    <button class="secondary" id="playSegment">Reproduzir corte selecionado</button>
    <button class="secondary" id="setStart">Usar tempo atual como Início</button>
    <button class="secondary" id="setEnd">Usar tempo atual como Fim</button>
  </div>
  <div class="row">
    <span class="status" id="webcamInfo">Webcam crop: automático</span>
    <button class="secondary" id="saveWebcam" style="display:none;">Salvar webcam.json</button>
  </div>
</section>
<section class="right" id="list"></section>
<script>
let state = { highlights: [], title: "", webcam_crop: null };
let activeIndex = 0;
let stopAt = null;
let cropMode = false;
let dragStart = null;
let pendingCrop = null;

const player = document.getElementById("player");
const canvas = document.getElementById("cropCanvas");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const listEl = document.getElementById("list");
const webcamInfo = document.getElementById("webcamInfo");
const saveWebcamBtn = document.getElementById("saveWebcam");

function setStatus(msg) { statusEl.textContent = msg; }

async function loadState() {
  const res = await fetch("/api/state");
  state = await res.json();
  document.getElementById("title").textContent = "Clipper Review — " + (state.title || "Vídeo");
  if (state.webcam_crop) {
    webcamInfo.textContent = "Webcam crop: " + state.webcam_crop.join(",");
  }
  renderList();
  setStatus(state.highlights.length + " corte(s) carregado(s)");
}

function renderList() {
  listEl.innerHTML = "";
  state.highlights.forEach((h, i) => {
    const card = document.createElement("div");
    card.className = "card" + (i === activeIndex ? " active" : "");
    const dur = Math.max(0, (Number(h.end) - Number(h.start))).toFixed(1);
    card.innerHTML = `
      <div class="row" style="justify-content:space-between;">
        <strong>#${i + 1} · <span class="badge">score ${h.score ?? 0}</span> · <span class="badge">${dur}s</span></strong>
        <div class="row">
          <button class="secondary" data-play="${i}">Play</button>
          <button class="danger" data-del="${i}">Remover</button>
        </div>
      </div>
      <div class="row">
        <label>Início (s)<input type="number" step="0.1" value="${h.start}" data-field="start" data-idx="${i}" /></label>
        <label>Fim (s)<input type="number" step="0.1" value="${h.end}" data-field="end" data-idx="${i}" /></label>
        <label>Score<input type="number" min="0" max="100" value="${h.score ?? 0}" data-field="score" data-idx="${i}" /></label>
      </div>
      <label style="font-size:12px;color:#9aa3b8;">Título<input type="text" value="${(h.title || "").replace(/"/g, "&quot;")}" data-field="title" data-idx="${i}" /></label>
      <label style="font-size:12px;color:#9aa3b8;">Hook (abertura)<input type="text" value="${(h.hook || "").replace(/"/g, "&quot;")}" data-field="hook" data-idx="${i}" /></label>
    `;
    card.addEventListener("click", (e) => {
      if (e.target.tagName !== "BUTTON") { activeIndex = i; renderList(); }
    });
    listEl.appendChild(card);
  });
  listEl.querySelectorAll("input").forEach((el) => {
    el.addEventListener("change", (e) => {
      const i = Number(e.target.dataset.idx);
      const field = e.target.dataset.field;
      state.highlights[i][field] = (field === "title" || field === "hook") ? e.target.value : Number(e.target.value);
      setStatus("Alterações não salvas");
    });
  });
  listEl.querySelectorAll("[data-play]").forEach((btn) => {
    btn.addEventListener("click", () => {
      activeIndex = Number(btn.dataset.play);
      playActive();
      renderList();
    });
  });
  listEl.querySelectorAll("[data-del]").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.highlights.splice(Number(btn.dataset.del), 1);
      activeIndex = Math.max(0, Math.min(activeIndex, state.highlights.length - 1));
      renderList();
      setStatus("Alterações não salvas");
    });
  });
}

function playActive() {
  const h = state.highlights[activeIndex];
  if (!h) return;
  player.currentTime = Number(h.start) || 0;
  stopAt = Number(h.end) || null;
  player.play();
}

player.addEventListener("timeupdate", () => {
  if (stopAt !== null && player.currentTime >= stopAt) {
    player.pause();
    stopAt = null;
  }
});

document.getElementById("playSegment").addEventListener("click", playActive);
document.getElementById("setStart").addEventListener("click", () => {
  if (!state.highlights[activeIndex]) return;
  state.highlights[activeIndex].start = Number(player.currentTime.toFixed(2));
  renderList();
  setStatus("Início atualizado (não salvo)");
});
document.getElementById("setEnd").addEventListener("click", () => {
  if (!state.highlights[activeIndex]) return;
  state.highlights[activeIndex].end = Number(player.currentTime.toFixed(2));
  renderList();
  setStatus("Fim atualizado (não salvo)");
});
document.getElementById("addClip").addEventListener("click", () => {
  const s = Number(player.currentTime.toFixed(2));
  state.highlights.push({ start: s, end: Number((s + 30).toFixed(2)), score: 80, title: "Novo corte", hook: "", reason: "Manual", llm: "manual" });
  activeIndex = state.highlights.length - 1;
  renderList();
  setStatus("Novo corte adicionado (não salvo)");
});
document.getElementById("saveBtn").addEventListener("click", async () => {
  const res = await fetch("/api/highlights", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(state.highlights) });
  if (res.ok) setStatus("highlights.json salvo!");
  else setStatus("Erro ao salvar: " + await res.text());
});

document.getElementById("toggleCrop").addEventListener("click", () => {
  cropMode = !cropMode;
  canvas.classList.toggle("active", cropMode);
  if (cropMode) {
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    player.pause();
    setStatus("Arraste sobre o vídeo para marcar a webcam");
  }
});

canvas.addEventListener("mousedown", (e) => {
  const r = canvas.getBoundingClientRect();
  dragStart = { x: e.clientX - r.left, y: e.clientY - r.top };
});
canvas.addEventListener("mousemove", (e) => {
  if (!dragStart) return;
  const r = canvas.getBoundingClientRect();
  const x = e.clientX - r.left, y = e.clientY - r.top;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#00e5ff";
  ctx.lineWidth = 2;
  ctx.strokeRect(dragStart.x, dragStart.y, x - dragStart.x, y - dragStart.y);
});
canvas.addEventListener("mouseup", (e) => {
  if (!dragStart || !player.videoWidth) return;
  const r = canvas.getBoundingClientRect();
  const x2 = e.clientX - r.left, y2 = e.clientY - r.top;
  const sx = player.videoWidth / canvas.width, sy = player.videoHeight / canvas.height;
  const rx = Math.max(0, Math.round(Math.min(dragStart.x, x2) * sx));
  const ry = Math.max(0, Math.round(Math.min(dragStart.y, y2) * sy));
  const rw = Math.max(2, Math.round(Math.abs(x2 - dragStart.x) * sx));
  const rh = Math.max(2, Math.round(Math.abs(y2 - dragStart.y) * sy));
  dragStart = null;
  pendingCrop = [rx - (rx % 2), ry - (ry % 2), rw - (rw % 2), rh - (rh % 2)];
  webcamInfo.textContent = "Novo webcam crop: " + pendingCrop.join(",");
  saveWebcamBtn.style.display = "inline-block";
});

saveWebcamBtn.addEventListener("click", async () => {
  if (!pendingCrop) return;
  const res = await fetch("/api/webcam", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ webcam_crop: pendingCrop }) });
  if (res.ok) {
    state.webcam_crop = pendingCrop;
    saveWebcamBtn.style.display = "none";
    setStatus("webcam.json salvo! Use clipper render --gameplay-split.");
  }
});

loadState();
</script>
</body>
</html>
"""


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        temporary = Path(tmp.name)
        json.dump(payload, tmp, ensure_ascii=False, indent=2)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _resolve_video(workdir: Path) -> tuple[Path | None, str]:
    meta_path = workdir / "source.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            video = Path(meta["video"])
            if video.is_file():
                return video, str(meta.get("title") or workdir.name)
        except (OSError, ValueError, KeyError):
            pass
    for suffix in (".mp4", ".mkv", ".webm", ".mov"):
        candidate = workdir / f"source{suffix}"
        if candidate.is_file():
            return candidate, workdir.name
    return None, workdir.name


class ReviewApp:
    def __init__(self, workdir: Path) -> None:
        self.workdir = workdir.resolve()
        if not self.workdir.is_dir():
            raise ValueError(f"workdir not found: {self.workdir}")
        self.video_path, self.title = _resolve_video(self.workdir)
        self.highlights_path = self.workdir / "highlights.json"
        self.webcam_path = self.workdir / "webcam.json"

    @staticmethod
    def _json_response(payload: object, status: int = HTTPStatus.OK) -> tuple[int, dict[str, str], bytes]:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return int(status), {"Content-Type": "application/json; charset=utf-8", "Content-Length": str(len(raw))}, raw

    def handle(self, method: str, path: str, headers: dict[str, str] | None = None,
               body: bytes = b"") -> tuple[int, dict[str, str], bytes]:
        route = urlparse(path).path
        req_headers = {k.lower(): v for k, v in (headers or {}).items()}
        if method == "GET":
            if route == "/":
                raw = HTML_PAGE.encode("utf-8")
                return int(HTTPStatus.OK), {"Content-Type": "text/html; charset=utf-8", "Content-Length": str(len(raw))}, raw
            if route == "/api/state":
                highlights = []
                if self.highlights_path.is_file():
                    highlights = json.loads(self.highlights_path.read_text(encoding="utf-8"))
                webcam_crop = None
                if self.webcam_path.is_file():
                    try:
                        webcam_crop = json.loads(self.webcam_path.read_text(encoding="utf-8")).get("webcam_crop")
                    except (OSError, ValueError):
                        webcam_crop = None
                return self._json_response({"title": self.title, "highlights": highlights, "webcam_crop": webcam_crop})
            if route == "/video":
                if self.video_path is None or not self.video_path.is_file():
                    return int(HTTPStatus.NOT_FOUND), {"Content-Type": "text/plain"}, b"Video file not found"
                size = self.video_path.stat().st_size
                range_header = req_headers.get("range")
                start, end = 0, size - 1
                status = HTTPStatus.OK
                if range_header:
                    match = _RANGE_RE.match(range_header.strip())
                    if match:
                        if match.group(1):
                            start = int(match.group(1))
                        if match.group(2):
                            end = min(size - 1, int(match.group(2)))
                        if start > end or start >= size:
                            return int(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE), {}, b""
                        status = HTTPStatus.PARTIAL_CONTENT
                length = end - start + 1
                with self.video_path.open("rb") as handle:
                    handle.seek(start)
                    data = handle.read(length)
                resp_headers = {
                    "Content-Type": "video/mp4",
                    "Accept-Ranges": "bytes",
                    "Content-Length": str(len(data)),
                }
                if status == HTTPStatus.PARTIAL_CONTENT:
                    resp_headers["Content-Range"] = f"bytes {start}-{end}/{size}"
                return int(status), resp_headers, data
            return int(HTTPStatus.NOT_FOUND), {}, b"Not Found"

        if method == "POST":
            try:
                parsed = json.loads(body.decode("utf-8"))
            except (UnicodeError, ValueError) as exc:
                return self._json_response({"error": f"invalid JSON: {exc}"}, HTTPStatus.BAD_REQUEST)
            if route == "/api/highlights":
                if not isinstance(parsed, list):
                    return self._json_response({"error": "highlights must be a JSON array"}, HTTPStatus.BAD_REQUEST)
                cleaned = []
                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    try:
                        s, e = float(item["start"]), float(item["end"])
                    except (KeyError, TypeError, ValueError):
                        return self._json_response({"error": "each clip needs numeric start and end"}, HTTPStatus.BAD_REQUEST)
                    if e <= s:
                        return self._json_response({"error": "end must be greater than start"}, HTTPStatus.BAD_REQUEST)
                    cleaned.append({
                        "start": round(s, 3),
                        "end": round(e, 3),
                        "score": max(0, min(100, int(item.get("score", 0)))),
                        "title": str(item.get("title", "")).strip(),
                        "hook": str(item.get("hook", "")).strip(),
                        "reason": str(item.get("reason", "")).strip(),
                        "llm": str(item.get("llm", "manual")),
                        "selection_score": item.get("selection_score"),
                        "speaker_focus": item.get("speaker_focus"),
                        "campaign_review": item.get("campaign_review"),
                    })
                _atomic_json(self.highlights_path, cleaned)
                return self._json_response({"saved": len(cleaned)})
            if route == "/api/webcam":
                crop = parsed.get("webcam_crop") if isinstance(parsed, dict) else None
                if not isinstance(crop, list) or len(crop) != 4:
                    return self._json_response({"error": "webcam_crop must be [x, y, w, h]"}, HTTPStatus.BAD_REQUEST)
                try:
                    x, y, w, h = (int(v) for v in crop)
                except (TypeError, ValueError):
                    return self._json_response({"error": "webcam_crop values must be integers"}, HTTPStatus.BAD_REQUEST)
                if min(x, y) < 0 or min(w, h) < 2:
                    return self._json_response({"error": "invalid crop bounds"}, HTTPStatus.BAD_REQUEST)
                if self.video_path is not None and self.video_path.is_file():
                    _save_cached_webcam(self.webcam_path, self.video_path, (x, y, w, h), manual=True)
                else:
                    _atomic_json(self.webcam_path, {"webcam_crop": [x, y, w, h], "manual": True})
                return self._json_response({"webcam_crop": [x, y, w, h]})
            return int(HTTPStatus.NOT_FOUND), {}, b"Not Found"
        return int(HTTPStatus.METHOD_NOT_ALLOWED), {}, b""


def create_server(workdir: Path, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    app = ReviewApp(workdir)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            return

        def _dispatch(self, method: str) -> None:
            length = int(self.headers.get("Content-Length") or "0")
            body = self.rfile.read(length) if length > 0 else b""
            route = urlparse(self.path).path
            if method == "GET" and route == "/video" and app.video_path and app.video_path.is_file():
                size = app.video_path.stat().st_size
                range_header = self.headers.get("Range")
                start, end = 0, size - 1
                status = HTTPStatus.OK
                if range_header:
                    match = _RANGE_RE.match(range_header.strip())
                    if match:
                        if match.group(1):
                            start = int(match.group(1))
                        if match.group(2):
                            end = min(size - 1, int(match.group(2)))
                        if start > end or start >= size:
                            self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                            return
                        status = HTTPStatus.PARTIAL_CONTENT
                total = end - start + 1
                self.send_response(status)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(total))
                if status == HTTPStatus.PARTIAL_CONTENT:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.end_headers()
                with app.video_path.open("rb") as handle:
                    handle.seek(start)
                    remaining = total
                    while remaining > 0:
                        chunk = handle.read(min(65536, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
                return
            status_code, resp_headers, payload = app.handle(method, self.path, dict(self.headers.items()), body)
            self.send_response(status_code)
            for k, v in resp_headers.items():
                self.send_header(k, v)
            self.end_headers()
            if payload:
                self.wfile.write(payload)

        def do_GET(self) -> None:
            self._dispatch("GET")

        def do_POST(self) -> None:
            self._dispatch("POST")

    return ThreadingHTTPServer((host, port), Handler)


def serve(workdir: Path, host: str = "127.0.0.1", port: int = 8765) -> None:
    server = create_server(workdir, host=host, port=port)
    actual_port = server.server_address[1]
    print(f"[review] open http://{host}:{actual_port} (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[review] stopped", flush=True)
    finally:
        server.server_close()
