"""Speaker preferences based on supplied, timed speaker labels."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from clipper.transcribe import Word


@dataclass(frozen=True)
class Turn:
    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class SpeakerPreference:
    speaker: str
    turns: tuple[Turn, ...]
    mode: str = "prefer"
    weight: float = 0.25
    label_source: str = "supplied_timestamps"

    @property
    def fingerprint(self) -> str:
        data = {"speaker": self.speaker.casefold(), "mode": self.mode, "weight": self.weight,
                "label_source": self.label_source,
                "turns": [asdict(t) for t in self.turns]}
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:10]

    def label(self, word: Word) -> str:
        matches = [(max(0.0, min(word.end, t.end) - max(word.start, t.start)), t.speaker)
                   for t in self.turns if t.start < word.end and t.end > word.start]
        overlap, speaker = max(matches, default=(0.0, "unknown"))
        return speaker if overlap > (word.end - word.start) / 2 and overlap > 0 else "unknown"

    def prompt(self) -> str:
        rule = ("Prefer moments where this participant makes a clear, compelling point. Keep questions and "
                "counterpoints when necessary to preserve context.") if self.mode == "prefer" else (
                "Return only clips whose spoken words are all assigned to this participant. "
                "Return fewer clips or none if no eligible moment exists.")
        return ("\n\nSpeaker preference for this batch: " + json.dumps(self.speaker, ensure_ascii=False)
                + ". " + rule
                + " Speaker labels are supplied as timestamp annotations; names follow the supplied mapping. "
                  "Do not identify a speaker by their opinions, by a name mentioned in the text, or by quoted words. "
                  "Unlabeled speech is unknown. Preserve the actual wording, order and meaning; "
                  "do not imply that another participant's words belong to the preferred speaker. "
                  "Keep score as an editorial/viral quality score independent of this preference; "
                  "the pipeline applies the preference weight separately.")

    def evaluate(self, words: list[Word], start: float, end: float) -> dict:
        preferred = other = total = 0.0
        for word in words:
            a, b = max(start, word.start), min(end, word.end)
            if b <= a:
                continue
            total += b - a
            for turn in self.turns:
                overlap = max(0.0, min(b, turn.end) - max(a, turn.start))
                if turn.speaker.casefold() == self.speaker.casefold():
                    preferred += overlap
                else:
                    other += overlap
        unknown = max(0.0, total - preferred - other)
        return {"speaker": self.speaker, "mode": self.mode, "weight": self.weight,
                "label_source": self.label_source, "timeline_fingerprint": self.fingerprint,
                "preferred_speech_share": round(preferred / total, 4) if total else 0.0,
                "other_speech_seconds": round(other, 4), "unknown_speech_seconds": round(unknown, 4),
                "eligible": total > 0 and (self.mode != "only" or (preferred > 0 and other <= 0.001 and unknown <= 0.001))}


def load_preference(path: Path, speaker: str, duration: float | None = None, mode: str = "prefer",
                    weight: float = 0.25) -> SpeakerPreference:
    if mode not in {"prefer", "only"} or not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("speaker mode must be prefer/only and weight must be between 0 and 1")
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("turns") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError("speaker timeline must be a list or an object with a turns list")
    turns = []
    limit = duration if duration is not None else math.inf
    for item in items:
        try:
            start, end = float(item["start"]), float(item["end"])
            name = item["speaker"]
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError("each speaker turn needs start, end and speaker") from exc
        if (not isinstance(name, str) or not name.strip() or name.strip().casefold() == "unknown"
                or not all(math.isfinite(v) for v in (start, end)) or not 0 <= start < limit
                or not start < end <= limit + 0.5):
            raise ValueError("speaker turns need valid names and finite times within the video")
        turns.append(Turn(start, min(end, limit), name.strip()))
    turns.sort(key=lambda t: (t.start, t.end))
    if any(current.start < previous.end for previous, current in zip(turns, turns[1:])):
        raise ValueError("speaker turns overlap; use unambiguous non-overlapping annotations")
    names = {t.speaker.casefold(): t.speaker for t in turns}
    if speaker.strip().casefold() not in names:
        raise ValueError(f"preferred speaker {speaker!r} is not present in the supplied timeline")
    source = data.get("source", "supplied_timestamps") if isinstance(data, dict) else "supplied_timestamps"
    return SpeakerPreference(names[speaker.strip().casefold()], tuple(turns), mode, weight, str(source))
