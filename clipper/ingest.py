import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from clipper.ffmpeg import ffmpeg_exe
from clipper.attention import save_source

VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov"}


@dataclass
class Source:
    video: Path
    title: str
    workdir: Path
    chat: Path | None
    engagement: Path | None = None


def is_url(value: str) -> bool:
    return bool(re.match(r"^https?://", value, re.IGNORECASE))


def slugify(text: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^\w]+", "-", text.lower(), flags=re.UNICODE).strip("-_")
    return slug[:max_len].strip("-_") or "video"


def fetch(source: str, out_root: Path, chat: str | None = None) -> Source:
    if is_url(source):
        result = _download(source, out_root)
    else:
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        workdir = out_root / slugify(path.stem)
        workdir.mkdir(parents=True, exist_ok=True)
        result = Source(video=path, title=path.stem, workdir=workdir, chat=None)
    if chat:
        target = result.workdir / f"chat{Path(chat).suffix or '.json'}"
        shutil.copyfile(chat, target)
        result.chat = target
    return result


def _download(url: str, out_root: Path) -> Source:
    import yt_dlp

    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "noplaylist": True}) as probe:
        info = probe.extract_info(url, download=False)
    title = info.get("title") or "video"
    workdir = out_root / slugify(f"{title}-{info.get('id', '')}")
    workdir.mkdir(parents=True, exist_ok=True)
    engagement = save_source(info, workdir, url)
    existing = _find_video(workdir)
    if existing is None:
        print(f"[ingest] downloading {title}", flush=True)
        opts = {
            "outtmpl": str(workdir / "source.%(ext)s"),
            "format": "bv*[height<=1080]+ba/b[height<=1080]/b",
            "merge_output_format": "mp4",
            "ffmpeg_location": ffmpeg_exe(),
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.process_ie_result(info.copy(), download=True)
        existing = _find_video(workdir)
        if existing is not None:
            print("[ingest] downloading optional live chat", flush=True)
            chat_opts = {
                **opts,
                "skip_download": True,
                "writesubtitles": True,
                "subtitleslangs": ["live_chat", "rechat"],
                "ignoreerrors": True,
                "no_warnings": False,
            }
            try:
                with yt_dlp.YoutubeDL(chat_opts) as ydl:
                    ydl.process_ie_result(info.copy(), download=True)
            except yt_dlp.utils.DownloadError as exc:
                print(f"[ingest] optional live chat unavailable; continuing without it: {exc}", flush=True)
    if existing is None:
        raise RuntimeError(f"download finished but no video found in {workdir}")
    chat = next((p for p in sorted(workdir.glob("source*.json")) if p.name != "source.json"), None)
    return Source(video=existing, title=title, workdir=workdir, chat=chat, engagement=engagement)


def _find_video(workdir: Path) -> Path | None:
    videos = [p for p in workdir.glob("source.*") if p.name.lower() in {f"source{s}" for s in VIDEO_SUFFIXES}]
    return max(videos, key=lambda p: p.stat().st_size) if videos else None
