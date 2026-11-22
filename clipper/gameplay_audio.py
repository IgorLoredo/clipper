"""Lightweight acoustic measurements for mixed gameplay audio."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from clipper.ffmpeg import FFmpegError, iter_audio_seconds
from clipper.transcribe import Transcript, Word

VERSION = 1
RATE = 8000


@dataclass(frozen=True)
class AudioEvent:
    start: float
    end: float
    peak_time: float
    event_type: str
    strength: float
    confidence: float | None
    features: dict[str, float]


def _pitch(frame: np.ndarray) -> tuple[float, float]:
    frame = frame - frame.mean()
    if np.sqrt(np.mean(frame ** 2)) < 60:
        return np.nan, 0.0
    spectrum = np.fft.rfft(frame, n=2 * len(frame))
    lag_min, lag_max = RATE // 450, RATE // 65
    lags = np.arange(lag_min - 1, lag_max + 2)
    ac = np.fft.irfft(abs(spectrum) ** 2)[lags]
    energy = np.r_[0, np.cumsum(frame ** 2)]
    ac /= np.sqrt(np.maximum(1, energy[len(frame) - lags] * (energy[-1] - energy[lags])))
    maxima = np.flatnonzero((ac[1:-1] > ac[:-2]) & (ac[1:-1] >= ac[2:])) + 1
    if not maxima.size:
        return np.nan, 0.0
    best = max(ac[maxima])
    if best < .72:
        return np.nan, 0.0
    peak = next(i for i in maxima if ac[i] >= max(.72, .9 * best))
    return RATE / lags[peak], float(ac[peak])


def _normalize(values: np.ndarray, floor: float) -> np.ndarray:
    baseline = values[np.isfinite(values)]
    if not baseline.size:
        return np.zeros_like(values)
    median = np.median(baseline)
    scale = max(floor, 1.4826 * np.median(abs(baseline - median)))
    return np.nan_to_num((values - median) / scale)


def analyze(chunks, words: list[Word], duration: float) -> list[AudioEvent]:
    """Use one-second buckets, sampling pitch only within transcribed speech."""
    rows = []
    valid_words = sorted((w for w in words if 0 <= w.start < w.end and w.start < duration), key=lambda w: w.start)
    starts = np.array([w.start for w in valid_words])
    ends = np.array([w.end for w in valid_words])
    # Whisper timestamps are ordered. Search by start and keep a small overlap for
    # words straddling the preceding bucket.
    previous_level = None
    for second, chunk in enumerate(chunks):
        if second >= duration or not len(chunk):
            break
        samples = np.asarray(chunk, dtype=float)
        blocks = [samples[i:i + RATE // 5] for i in range(0, len(samples), RATE // 5)]
        levels = [20 * np.log10(max(1, np.sqrt(np.mean(block ** 2)))) for block in blocks]
        rises = np.diff(levels, prepend=levels[0] if previous_level is None else previous_level)
        peak = int(np.argmax(rises))
        previous_level = levels[-1]
        left, right = np.searchsorted(starts, [max(0, second - 2), second + 1])
        f0 = []
        for offset in (.1, .3, .5, .7, .9):
            middle = second + offset
            if not np.any((starts[left:right] <= middle) & (ends[left:right] >= middle)):
                continue
            center = round(offset * RATE)
            frame = samples[center - 320:center + 320]
            if len(frame) == 640:
                hz, confidence = _pitch(frame)
                if confidence:
                    f0.append((hz, confidence))
        pitch, confidence = np.median(f0, axis=0) if len(f0) >= 2 else (np.nan, 0.0)
        words_in_bucket = np.searchsorted(starts, second + 1) - np.searchsorted(starts, second)
        rows.append((max(levels), max(0, rises[peak]), peak / 5, pitch, confidence, words_in_bucket))
    if not rows:
        return []
    level, rise, peak_offset, pitch, confidence, counts = np.asarray(rows).T
    if not np.any(level > 36):
        return []
    semitones = 12 * np.log2(pitch)
    pitch_shift = np.r_[np.nan, abs(np.diff(semitones))]
    pace = np.convolve(np.pad(counts, (1, 1), mode="edge"), np.ones(3) / 3, mode="valid")
    pace_change = abs(pace - np.median(pace[pace > 0])) if np.any(pace > 0) else pace
    gaps = np.zeros(len(level))
    word_gaps = starts[1:] - ends[:-1]
    normal_gap = np.median(word_gaps[word_gaps >= 0]) if np.any(word_gaps >= 0) else 0.0
    for before, after, gap in zip(ends[:-1], starts[1:], word_gaps):
        if not max(.65, normal_gap * 2) <= gap <= 4:
            continue
        interior = level[int(np.ceil(before)):int(after)]
        if interior.size and np.min(interior) > np.median(level) - 4:
            continue
        for boundary in (before, after):
            index = min(len(gaps) - 1, int(boundary))
            gaps[index] = max(gaps[index], gap)
    onset = _normalize(rise, 3.0)
    intensity = _normalize(level, 4.0)
    pitch_z = _normalize(pitch_shift, 1.5)
    pace_z = _normalize(np.where(pace > 0, pace_change, np.nan), .4)
    # Correlated descriptors share one strength. Timing alone is retained as
    # context but cannot imply expressive audio.
    scores = np.stack((np.maximum(onset, intensity), pitch_z, pace_z, gaps / .65))
    kinds = ("intensity_change", "pitch_change", "speech_rate_change", "speech_pause")
    events = []
    for i in np.flatnonzero(scores.max(axis=0) >= 2.5):
        kind = kinds[int(np.argmax(scores[:, i]))]
        features = {"level_dbfs": round(level[i] - 20 * np.log10(32768), 2),
                    "rise_db": round(rise[i], 2), "onset_z": round(onset[i], 2),
                    "intensity_z": round(intensity[i], 2), "words_per_second": round(pace[i], 2),
                    "pace_z": round(pace_z[i], 2), "pause_seconds": round(gaps[i], 2)}
        if np.isfinite(pitch[i]):
            features.update(pitch_hz=round(pitch[i], 1), pitch_confidence=round(confidence[i], 3))
        if np.isfinite(pitch_shift[i]):
            features.update(pitch_shift_semitones=round(pitch_shift[i], 2), pitch_z=round(pitch_z[i], 2))
        if kind == "pitch_change":
            reliability = round(min(confidence[i], confidence[i - 1]), 3)
        else:
            reliability = None
        peak = i + (peak_offset[i] if kind == "intensity_change" else .5)
        end = min(duration, float(i + 1))
        peak = min(float(peak), end - min(.001, (end - i) / 2))
        events.append(AudioEvent(float(i), end, peak,
                                 kind, round(min(10.0, scores[:, i].max()), 2), reliability, features))
    return events


def collect(video: Path, transcript: Transcript, cache: Path) -> list[AudioEvent] | None:
    stat = video.stat()
    key = {"version": VERSION, "video": str(video.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
           "transcript": hashlib.sha256(repr(transcript).encode()).hexdigest()}
    if cache.is_file():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if data["key"] == key:
                return [AudioEvent(**event) for event in data["events"]]
        except (OSError, ValueError, KeyError, TypeError):
            pass
    try:
        events = analyze(iter_audio_seconds(video, RATE, check=True, align=True), transcript.words, transcript.duration)
    except (FFmpegError, OSError) as exc:
        print(f"[gameplay] acoustic analysis unavailable: {exc}; using existing signals", flush=True)
        return None
    report = {"key": key, "laughter_detection": "unavailable", "events": [asdict(event) for event in events]}
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=cache.parent, delete=False) as file:
        temporary = Path(file.name)
    try:
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(cache)
    finally:
        temporary.unlink(missing_ok=True)
    return events
