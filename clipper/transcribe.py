import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

from clipper import ffmpeg
from clipper.config import Settings


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Transcript:
    language: str
    duration: float
    words: list[Word] = field(default_factory=list)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Transcript":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            language=data["language"],
            duration=data["duration"],
            words=[Word(**w) for w in data["words"]],
        )

    def words_between(self, start: float, end: float) -> list[Word]:
        return [w for w in self.words if w.end > start and w.start < end]


def find_json3(workdir: Path, language: str | None = None) -> Path | None:
    files = sorted(workdir.glob("*.json3"))
    if not files:
        return None
    if language:
        prefix = language.lower().split("-", 1)[0]
        for candidate in files:
            name = candidate.name.lower()
            if f".{prefix}" in name or name.startswith(prefix):
                return candidate
    return max(files, key=lambda p: p.stat().st_size)


def load_json3(path: Path, duration: float | None = None, language: str = "pt") -> Transcript:
    """Parse a YouTube json3 subtitle file into word-timed Transcript."""
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    events = data.get("events") or []
    words: list[Word] = []
    max_end = 0.0
    for idx, event in enumerate(events):
        segs = event.get("segs")
        if not segs or "tStartMs" not in event:
            continue
        base_ms = float(event["tStartMs"])
        ev_dur_ms = float(event.get("dDurationMs") or 0.0)
        next_start_ms = None
        for nxt in events[idx + 1:]:
            if nxt.get("segs") and "tStartMs" in nxt:
                next_start_ms = float(nxt["tStartMs"])
                break
        if ev_dur_ms <= 0 and next_start_ms is not None and next_start_ms > base_ms:
            ev_dur_ms = min(next_start_ms - base_ms, 5000.0)
        ev_end_ms = base_ms + max(ev_dur_ms, 300.0)
        if next_start_ms is not None and next_start_ms > base_ms:
            ev_end_ms = min(ev_end_ms, next_start_ms)

        timed_segs: list[tuple[float, str]] = []
        for seg in segs:
            raw = str(seg.get("utf8", "")).replace("\n", " ").strip()
            if not raw:
                continue
            offset_ms = float(seg.get("tOffsetMs") or 0.0)
            timed_segs.append((base_ms + offset_ms, raw))
        if not timed_segs:
            continue
        for pos, (seg_start_ms, text) in enumerate(timed_segs):
            seg_end_ms = timed_segs[pos + 1][0] if pos + 1 < len(timed_segs) else ev_end_ms
            if seg_end_ms <= seg_start_ms:
                seg_end_ms = seg_start_ms + 250.0
            tokens = [tok for tok in text.split() if tok]
            if not tokens:
                continue
            step = (seg_end_ms - seg_start_ms) / len(tokens)
            for k, token in enumerate(tokens):
                w_start = round((seg_start_ms + k * step) / 1000.0, 2)
                w_end = round(max(w_start + 0.05, (seg_start_ms + (k + 1) * step) / 1000.0), 2)
                words.append(Word(text=token, start=w_start, end=w_end))
                if w_end > max_end:
                    max_end = w_end

    total_duration = float(duration) if duration and duration > 0 else max(max_end, 1.0)
    return Transcript(language=language, duration=total_duration, words=words)


def _pick_device(settings: Settings) -> tuple[str, str, str]:
    device = settings.whisper_device
    if not device:
        try:
            import ctranslate2

            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    model = settings.whisper_model or ("large-v3-turbo" if device == "cuda" else "small")
    return device, compute_type, model


def transcribe(video: Path, cache: Path, settings: Settings, language: str | None = None, *,
               source_mode: str = "whisper", json3_path: Path | None = None) -> Transcript:
    if cache.is_file():
        return Transcript.load(cache)
    if source_mode in {"youtube", "auto"}:
        sub_file = json3_path or find_json3(cache.parent, language)
        if sub_file and sub_file.is_file():
            try:
                duration = ffmpeg.probe_duration(video) if video.is_file() else None
            except ffmpeg.FFmpegError:
                duration = None
            transcript = load_json3(sub_file, duration=duration, language=language or "pt")
            if transcript.words:
                print(f"[transcribe] loaded {len(transcript.words)} words from {sub_file.name}", flush=True)
                transcript.save(cache)
                return transcript
            if source_mode == "youtube":
                raise ValueError(f"YouTube subtitle file {sub_file} has no words")
        elif source_mode == "youtube":
            raise ValueError(f"no YouTube .json3 subtitles found in {cache.parent}")
    from faster_whisper import WhisperModel

    device, compute_type, model_name = _pick_device(settings)
    print(f"[transcribe] whisper={model_name} device={device}", flush=True)
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    segments, info = model.transcribe(
        str(video), language=language, word_timestamps=True, vad_filter=True,
    )
    words: list[Word] = []
    for segment in segments:
        for w in segment.words or []:
            text = w.word.strip()
            if text:
                words.append(Word(text=text, start=round(w.start, 2), end=round(w.end, 2)))
        print(f"\r[transcribe] {segment.end / max(info.duration, 1):6.1%}", end="", flush=True)
    print(flush=True)
    transcript = Transcript(language=info.language, duration=info.duration, words=words)
    transcript.save(cache)
    return transcript


def transcribe_range(video: Path, start: float, end: float, settings: Settings,
                     language: str | None = None, model=None) -> list[Word]:
    """Transcribe only a single clip interval [start, end] for fast caption refinement."""
    if end <= start:
        return []
    if model is None:
        from faster_whisper import WhisperModel

        device, compute_type, model_name = _pick_device(settings)
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
    with TemporaryDirectory() as tmp:
        wav = Path(tmp) / "clip.wav"
        ffmpeg.run(["-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(video),
                    "-vn", "-ac", "1", "-ar", "16000", str(wav)])
        segments, _ = model.transcribe(str(wav), language=language, word_timestamps=True, vad_filter=True)
        words: list[Word] = []
        for segment in segments:
            for w in segment.words or []:
                text = w.word.strip()
                if text:
                    words.append(Word(
                        text=text,
                        start=round(start + w.start, 2),
                        end=round(start + w.end, 2),
                    ))
    return words
