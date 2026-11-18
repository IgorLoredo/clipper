import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from clipper.ffmpeg import iter_audio_seconds


@dataclass
class Peak:
    time: float
    strength: float
    kind: str


def robust_z(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    median = np.median(values)
    mad = np.median(np.abs(values - median)) * 1.4826
    if mad < 1e-9:
        mad = float(np.std(values)) or 1.0
    return (values - median) / mad


def find_peaks(levels: np.ndarray, bucket_seconds: float, kind: str,
               threshold: float = 2.5, min_gap: float = 20.0, limit: int = 40) -> list[Peak]:
    z = robust_z(np.asarray(levels, dtype=float))
    candidates = sorted(
        ((float(z[i]), i * bucket_seconds) for i in range(len(z)) if z[i] >= threshold),
        reverse=True,
    )
    chosen: list[Peak] = []
    for strength, time in candidates:
        if all(abs(time - p.time) >= min_gap for p in chosen):
            chosen.append(Peak(time=time, strength=round(strength, 2), kind=kind))
        if len(chosen) >= limit:
            break
    return sorted(chosen, key=lambda p: p.time)


def audio_levels(video: Path) -> np.ndarray:
    levels = []
    for samples in iter_audio_seconds(video):
        rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2)) if samples.size else 0.0
        levels.append(20 * np.log10(rms + 1e-6))
    return np.array(levels)


def parse_chat_times(path: Path) -> list[float]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return _parse_jsonl_chat(text)
    if isinstance(data, dict) and "comments" in data:
        return [float(c["content_offset_seconds"]) for c in data["comments"] if "content_offset_seconds" in c]
    if isinstance(data, dict) and "replayChatItemAction" in data:
        return _parse_jsonl_chat(text)
    if isinstance(data, list):
        return [t for t in (_time_of(item) for item in data) if t is not None]
    return []


def _parse_jsonl_chat(text: str) -> list[float]:
    times = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        replay = obj.get("replayChatItemAction", {})
        offset = replay.get("videoOffsetTimeMsec")
        if offset is not None:
            actions = replay.get("actions")
            count = sum("addChatItemAction" in action for action in actions) if isinstance(actions, list) else 1
            times.extend([int(offset) / 1000] * count)
    return times


def _time_of(item) -> float | None:
    if isinstance(item, (int, float)):
        return float(item)
    if isinstance(item, dict):
        for key in ("t", "time", "offset", "content_offset_seconds"):
            if key in item:
                return float(item[key])
    return None


def chat_density(times: list[float], duration: float, bucket_seconds: float = 10.0) -> np.ndarray:
    buckets = max(1, int(np.ceil(duration / bucket_seconds)))
    counts = np.zeros(buckets)
    for t in times:
        idx = int(t // bucket_seconds)
        if 0 <= idx < buckets:
            counts[idx] += 1
    return counts


def collect(video: Path, duration: float, chat: Path | None, cache: Path) -> list[Peak]:
    if cache.is_file():
        return [Peak(**p) for p in json.loads(cache.read_text(encoding="utf-8"))]
    print("[signals] measuring audio energy", flush=True)
    peaks = find_peaks(audio_levels(video), 1.0, "audio")
    if chat and chat.is_file():
        times = parse_chat_times(chat)
        print(f"[signals] {len(times)} chat messages", flush=True)
        if times:
            peaks += find_peaks(chat_density(times, duration), 10.0, "chat", threshold=2.0)
    peaks.sort(key=lambda p: p.time)
    cache.write_text(json.dumps([asdict(p) for p in peaks]), encoding="utf-8")
    return peaks
