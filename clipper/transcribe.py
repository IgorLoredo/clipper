import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

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


def transcribe(video: Path, cache: Path, settings: Settings, language: str | None = None) -> Transcript:
    if cache.is_file():
        return Transcript.load(cache)
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
