"""Public replay popularity and local engagement signals; no LLM calls."""

import hashlib
import json
import math
from bisect import bisect_left
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from clipper.signals import Peak, chat_density, find_peaks, parse_chat_times


@dataclass(frozen=True)
class ReplayInterval:
    start: float
    end: float
    value: float


def normalize_replay(items: list) -> list[ReplayInterval]:
    result = []
    for item in items:
        try:
            start, end, value = (float(item[key]) for key in ("start_time", "end_time", "value"))
        except (KeyError, TypeError, ValueError):
            continue
        if all(math.isfinite(v) for v in (start, end, value)) and 0 <= start < end and 0 <= value <= 1:
            result.append(ReplayInterval(start, end, value))
    # Overlapping/duplicate bands must not inflate interval averages or coverage.
    valid = []
    for band in sorted(set(result), key=lambda b: (b.start, b.end)):
        if not valid or band.start >= valid[-1].end:
            valid.append(band)
    return valid


def save_source(info: dict, workdir: Path, url: str) -> Path:
    path = workdir / "engagement.json"
    bands = normalize_replay(info.get("heatmap") or [])
    if not bands and path.is_file():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("video_id") == info.get("id") and old.get("heatmap"):
            return path
    data = {
        "version": 1, "source": "youtube_most_replayed",
        "video_id": info.get("id"), "url": info.get("webpage_url") or url,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "available": bool(bands),
        "meaning": "Relative replay popularity (0-1); not absolute views, unique viewers or retention.",
        "heatmap": [{"start_time": b.start, "end_time": b.end, "value": b.value} for b in bands],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def refresh_source(url: str, workdir: Path) -> Path:
    import yt_dlp

    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "noplaylist": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    return save_source(info, workdir, url)


@dataclass
class Attention:
    duration: float
    replay: list[ReplayInterval]
    chat_times: list[float]
    chat_available: bool
    audio_peaks: list[Peak]

    @property
    def fingerprint(self) -> str:
        if not self.replay:
            return ""
        text = json.dumps([asdict(b) for b in self.replay], sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()[:10]

    def replay_peaks(self, limit: int = 12) -> list[Peak]:
        if len(self.replay) < 2:
            return []
        values = [b.value for b in self.replay]
        if max(values) - min(values) < 1e-6:
            return []
        threshold = max(0.5, float(np.percentile(values, 80)))
        candidates = []
        for i, band in enumerate(self.replay):
            left = values[i - 1] if i else -1
            right = values[i + 1] if i + 1 < len(values) else -1
            if band.value >= threshold and band.value >= max(left, right):
                candidates.append(band)
        chosen = []
        for band in sorted(candidates, key=lambda b: b.value, reverse=True):
            center = (band.start + band.end) / 2
            if all(abs(center - p.time) >= 30 for p in chosen):
                chosen.append(Peak(center, round(band.value, 4), "replay"))
            if len(chosen) >= limit:
                break
        return sorted(chosen, key=lambda p: p.time)

    def chat_peaks(self) -> list[Peak]:
        if not self.chat_available or not self.chat_times:
            return []
        return find_peaks(chat_density(self.chat_times, self.duration), 10.0, "chat", threshold=2.0)

    def interval(self, start: float, end: float) -> dict:
        start, end = max(0.0, start), min(self.duration, end)
        if end <= start:
            raise ValueError("metric interval must have a positive duration within the video")
        overlaps = [(min(end, b.end) - max(start, b.start), b.value) for b in self.replay
                    if b.end > start and b.start < end]
        covered = sum(length for length, _ in overlaps)
        messages = bisect_left(self.chat_times, end) - bisect_left(self.chat_times, start)
        audio = [p for p in self.audio_peaks if start <= p.time < end]
        return {
            "replay_index_mean": round(sum(length * value for length, value in overlaps) / covered, 4) if covered else None,
            "replay_index_max": round(max(value for _, value in overlaps), 4) if covered else None,
            "replay_coverage": round(covered / (end - start), 4),
            "chat_messages": messages if self.chat_available else None,
            "chat_messages_per_minute": round(messages * 60 / (end - start), 2) if self.chat_available else None,
            "audio_peak_count": len(audio),
            "audio_peak_strength_max": max((p.strength for p in audio), default=None),
        }

    def report(self, workdir: Path) -> dict:
        clips = []
        for path in sorted((workdir / "clips").glob("*/clips.json")):
            for clip in json.loads(path.read_text(encoding="utf-8")):
                clips.append({"batch": path.parent.name, "file": clip["file"],
                              "start": clip["start"], "end": clip["end"],
                              "metrics": self.interval(clip["start"], clip["end"])})
        spans = [(b.start, min(b.end, self.duration)) for b in self.replay if b.start < self.duration]
        if not spans:
            spans = [(float(t), min(t + 30.0, self.duration)) for t in range(0, math.ceil(self.duration), 30)]
        metric = "replay_index_mean" if self.replay else (
            "chat_messages_per_minute" if self.chat_available else (
                "audio_peak_strength_max" if self.audio_peaks else None))
        segments = []
        for start, end in spans:
            row = {"start": start, "end": end, **self.interval(start, end)}
            row["overlaps_existing_clip"] = any(c["start"] < end and c["end"] > start for c in clips)
            segments.append(row)
        ranked = sorted((s for s in segments if s[metric] is not None and s[metric] > 0),
                        key=lambda s: s[metric], reverse=True) if metric else []
        return {
            "version": 1, "duration": self.duration, "replay_available": bool(self.replay),
            "chat_available": self.chat_available, "ranking_metric": metric,
            "limitations": [
                "Replay index is relative popularity, not a count of views or unique viewers.",
                "Replay data has the source's time resolution; it is not measured per second.",
                "Chat messages and audio peaks are proxies, not audience retention.",
                "These signals support editorial judgment; they do not prove clip quality.",
            ],
            "replay_intervals": [asdict(b) for b in self.replay],
            "selection_hints": [asdict(p) for p in self.replay_peaks()],
            "segments": segments, "top_segments": ranked[:10],
            "top_uncovered_segments": [s for s in ranked if not s["overlaps_existing_clip"]][:10],
            "clips": clips,
        }


def load(duration: float, peaks: list[Peak], chat: Path | None, replay_path: Path | None) -> Attention:
    replay = []
    if replay_path and replay_path.is_file():
        data = json.loads(replay_path.read_text(encoding="utf-8"))
        replay = normalize_replay(data.get("heatmap") or [])
    available = bool(chat and chat.is_file())
    times = sorted(t for t in parse_chat_times(chat) if math.isfinite(t) and 0 <= t < duration) if available else []
    return Attention(duration, replay, times, available, [p for p in peaks if p.kind == "audio"])


def timestamp(seconds: float) -> str:
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def save_report(context: Attention, workdir: Path) -> Path:
    report = context.report(workdir)
    path = workdir / "attention.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Atenção por trecho", "",
             "O índice de replay é relativo (0 a 1). Não representa visualizações absolutas ou retenção.",
             "Chat e áudio são sinais auxiliares. O ranking não comprova a qualidade de um corte.", "",
             f"Replay público disponível: {'sim' if report['replay_available'] else 'não'}.", "",
             "## Trechos com maior sinal de atenção", "",
             "| Intervalo | Replay médio | Mensagens no chat | Já usado em corte |",
             "|---|---:|---:|---|"]
    for row in report["top_segments"]:
        replay = f"{row['replay_index_mean']:.3f}" if row['replay_index_mean'] is not None else "indisponível"
        chat = str(row['chat_messages']) if row['chat_messages'] is not None else "indisponível"
        lines.append(f"| {timestamp(row['start'])}–{timestamp(row['end'])} | {replay} | {chat} | "
                     + ("sim" if row['overlaps_existing_clip'] else "não") + " |")
    lines += ["", "## Regiões com sinal alto ainda não usadas", ""]
    if not report["top_uncovered_segments"]:
        lines.append("Nenhuma região adicional com sinal disponível foi encontrada.")
    for row in report["top_uncovered_segments"][:5]:
        lines.append(f"- {timestamp(row['start'])}–{timestamp(row['end'])}")
    lines += ["", "## Cortes existentes", "", "| Lote / arquivo | Replay médio | Mensagens no chat |",
              "|---|---:|---:|"]
    for clip in report["clips"]:
        metrics = clip["metrics"]
        replay = f"{metrics['replay_index_mean']:.3f}" if metrics['replay_index_mean'] is not None else "indisponível"
        chat = str(metrics['chat_messages']) if metrics['chat_messages'] is not None else "indisponível"
        lines.append(f"| {clip['batch']}/{clip['file']} | {replay} | {chat} |")
    (workdir / "attention.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def analyze(workdir: Path, url: str | None = None) -> Path:
    from clipper.transcribe import Transcript

    if url:
        refresh_source(url, workdir)
    transcript = Transcript.load(workdir / "transcript.json")
    signals = workdir / "signals.json"
    peaks = [Peak(**p) for p in json.loads(signals.read_text(encoding="utf-8"))] if signals.is_file() else []
    candidates = sorted(workdir.glob("source*.json")) + sorted(workdir.glob("chat*.json"))
    chat = next((p for p in candidates if p.name != "source.json"), None)
    context = load(transcript.duration, peaks, chat, workdir / "engagement.json")
    path = save_report(context, workdir)
    report = json.loads(path.read_text(encoding="utf-8"))
    print(f"[metrics] replay bands: {len(context.replay)}; existing clips: {len(report['clips'])}", flush=True)
    for row in report["top_uncovered_segments"][:3]:
        print(f"[metrics] unused region {timestamp(row['start'])}-{timestamp(row['end'])}: "
              f"{report['ranking_metric']}={row[report['ranking_metric']]}", flush=True)
    return path
