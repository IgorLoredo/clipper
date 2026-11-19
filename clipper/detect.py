import json
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

from clipper.codex_exchange import CodexResponseRequired
from clipper import gameplay_selection as gameplay
from clipper.llm import JsonLLM, LLMRefusal, ProvidersExhausted
from clipper.signals import Peak
from clipper.transcribe import Transcript, Word
from clipper.speakers import SpeakerPreference

SENTENCE_END = (".", "!", "?", "…")

SYSTEM_PROMPT = """You are an elite short-form video editor. You have studied thousands of viral clips on TikTok, YouTube Shorts and Instagram Reels and you know what makes a viewer stop scrolling, watch to the end and share.

You receive a time-stamped transcript excerpt from a long video or a live stream. Each line starts with its start time in seconds, like [123.4].

Find the moments with the highest viral potential. Rank them by these signals:
1. Hook: a line that creates instant curiosity in the first 3 seconds.
2. Emotional peak: real surprise, laughter, anger, vulnerability or hype.
3. Strong opinion: a polarizing or counter-intuitive statement.
4. Revelation: a surprising fact, number or confession.
5. Conflict: disagreement, pushback, a problem confronted head-on.
6. Quotable line: a sentence that works on its own as a quote.
7. Story payoff: the climax or twist of an anecdote.
8. Practical value: a concrete tip the viewer can apply right away.

Rules:
- Each clip must start on its hook and must not open mid-sentence or mid-thought.
- Each clip must be understandable without the rest of the video and end on a complete thought.
- Duration between {min_dur} and {max_dur} seconds.
- Clips must not overlap each other.
- Score 0-100 for viral potential only, not general quality. Be strict: most moments are below 60.
- Return at most {per_chunk} clips. Return fewer, or none, if the excerpt has nothing strong.
- Write "title" and "hook" in the transcript's language ({language}). The title is a short social media caption, max 70 characters; the hook has at most 16 words.
- "reason" explains in at most 20 words why the clip would perform."""

CONTENT_GUIDANCE = {
    "anime": """\nAnime-specific editorial rules:
- Prefer self-contained dialogue, a reveal, an emotional turning point, a memorable line or a clear story payoff.
- Do not invent visual events from the transcript. Avoid a clip that needs unseen context to make sense.
- Keep the original meaning; do not use misleading titles or fabricated spoilers.\n""",
    "mass": """\nMass-specific editorial rules:
- Prefer a complete reflection from the homily, Gospel explanation, prayer or practical faith message.
- Be reverent and accurate: never turn a sacred rite, personal prayer or scripture into rage bait or a misleading claim.
- Keep enough context to preserve the intended religious meaning and avoid cutting prayers or sentences in half.\n""",
}

SCHEMA = {
    "type": "object",
    "properties": {
        "highlights": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "score": {"type": "integer"},
                    "title": {"type": "string"},
                    "hook": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["start", "end", "score", "title", "hook", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["highlights"],
    "additionalProperties": False,
}


@dataclass
class Highlight:
    start: float
    end: float
    score: int
    title: str
    hook: str
    reason: str
    llm: str = ""
    selection_score: float | None = None
    speaker_focus: dict | None = None
    campaign_review: dict | None = None

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def ranking_score(self) -> float:
        return self.selection_score if self.selection_score is not None else float(self.score)


def windows(duration: float, size: float = 900.0, overlap: float = 60.0) -> list[tuple[float, float]]:
    if duration <= size:
        return [(0.0, duration)]
    spans, start = [], 0.0
    while start < duration:
        end = min(start + size, duration)
        spans.append((start, end))
        if end >= duration:
            break
        start = end - overlap
    return spans


def format_lines(words: list[Word], max_gap: float = 1.2, max_words: int = 25,
                 preference: SpeakerPreference | None = None) -> str:
    lines, current = [], []
    current_speaker = None
    for word in words:
        speaker = preference.label(word) if preference else None
        if current and (word.start - current[-1].end > max_gap or len(current) >= max_words
                        or speaker != current_speaker):
            lines.append(current)
            current = []
        current.append(word)
        current_speaker = speaker
        if word.text.endswith(SENTENCE_END):
            lines.append(current)
            current = []
    if current:
        lines.append(current)
    return "\n".join(f"[{line[0].start:.1f}] "
                     + (f"[speaker={json.dumps(preference.label(line[0]), ensure_ascii=False)}] " if preference else "")
                     + " ".join(w.text for w in line) for line in lines)


def format_hints(peaks: list[Peak], start: float, end: float) -> str:
    inside = [p for p in peaks if start <= p.time <= end]
    if not inside:
        return ""
    parts = []
    for kind, label in (("audio", "audio energy spikes (shouting, laughter, hype)"), ("chat", "chat activity spikes")):
        times = [f"{p.time:.0f}s" for p in inside if p.kind == kind]
        if times:
            parts.append(f"- {label}: {', '.join(times)}")
    replay = [p for p in inside if p.kind == "replay"]
    if replay:
        parts.append("- public replay popularity (relative index 0-1, NOT view counts; broad regions): "
                     + ", ".join(f"~{p.time:.0f}s (index {p.strength:.2f})" for p in replay))
    return "Extra signals measured in this excerpt. They often mark strong moments, but judge by content:\n" + "\n".join(parts)


def snap(start: float, end: float, words: list[Word], min_dur: float, max_dur: float,
         extend_limit: float = 4.0) -> tuple[float, float] | None:
    inside = [w for w in words if w.end > start - 0.3 and w.start < end + extend_limit]
    if not inside:
        return None
    first = next((i for i, w in enumerate(inside) if w.start >= start - 0.3), None)
    if first is None:
        return None
    within = [i for i in range(first, len(inside)) if inside[i].end <= end + 0.3]
    if not within:
        return None
    last = _extend_to_sentence_end(inside, within[-1], end + extend_limit)
    snapped_start = max(0.0, inside[first].start - 0.15)
    snapped_end = inside[last].end + 0.25
    if snapped_end - snapped_start > max_dur:
        fits = [w for w in inside[first:last + 1] if w.end + 0.25 - snapped_start <= max_dur]
        if not fits:
            return None
        snapped_end = fits[-1].end + 0.25
    if snapped_end - snapped_start < min_dur:
        return None
    return round(snapped_start, 2), round(snapped_end, 2)


def _extend_to_sentence_end(words: list[Word], last: int, limit: float) -> int:
    if words[last].text.endswith(SENTENCE_END):
        return last
    for i in range(last + 1, len(words)):
        if words[i].end > limit:
            break
        if words[i].text.endswith(SENTENCE_END):
            return i
    return last


def overlap_ratio(a: Highlight, b: Highlight) -> float:
    inter = min(a.end, b.end) - max(a.start, b.start)
    if inter <= 0:
        return 0.0
    return inter / min(a.duration, b.duration)


def dedupe(highlights: list[Highlight], max_overlap: float = 0.3) -> list[Highlight]:
    kept: list[Highlight] = []
    for h in sorted(highlights, key=lambda x: x.ranking_score, reverse=True):
        if all(overlap_ratio(h, k) < max_overlap for k in kept):
            kept.append(h)
    return kept


def parse_highlights(payload: dict) -> list[Highlight]:
    result = []
    for item in payload.get("highlights", []):
        try:
            start, end = float(item["start"]), float(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start:
            continue
        result.append(Highlight(
            start=start, end=end, score=max(0, min(100, int(item.get("score", 0)))),
            title=str(item.get("title", "")).strip(), hook=str(item.get("hook", "")).strip(),
            reason=str(item.get("reason", "")).strip(),
        ))
    return result


def progress_path(cache: Path) -> Path:
    return cache.with_name(f"{cache.stem}.progress.json")


def _load_progress(path: Path, params: dict) -> dict[str, dict]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("params") != params:
        print(f"[detect] {path.name} was made with other settings, starting over", flush=True)
        return {}
    return data.get("windows", {})


def _save_progress(path: Path, params: dict, done: dict[str, dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"params": params, "windows": done}, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _window_key(start: float, end: float) -> str:
    return f"{start:.1f}-{end:.1f}"


def _ask_window(llm: JsonLLM, system: str, user: str, index: int, gameplay_batch=False) -> dict:
    try:
        payload = llm(system, user, gameplay.SCHEMA if gameplay_batch else SCHEMA)
    except LLMRefusal as exc:
        print(f"[detect] chunk {index} refused: {exc}", flush=True)
        return {"llm": getattr(llm, "label", ""), "refused": True, "highlights": []}
    label = getattr(llm, "label", "")
    if gameplay_batch:
        return {"llm": label, "reviews": payload["reviews"]}
    found = [asdict(h) | {"llm": label} for h in parse_highlights(payload)]
    return {"llm": label, "highlights": found}


def _snap_all(done: dict[str, dict], keys: list[str], transcript: Transcript,
              min_dur: float, max_dur: float, preference: SpeakerPreference | None = None) -> list[Highlight]:
    found = []
    for key in keys:
        for item in done.get(key, {}).get("highlights", []):
            h = Highlight(**item)
            bounds = snap(h.start, h.end, transcript.words, min_dur, max_dur)
            if bounds:
                h.start, h.end = bounds
                if preference:
                    h.speaker_focus = preference.evaluate(transcript.words, h.start, h.end)
                    if not h.speaker_focus["eligible"]:
                        continue
                    h.selection_score = round((1 - preference.weight) * h.score
                                               + preference.weight * 100 * h.speaker_focus["preferred_speech_share"], 2)
                found.append(h)
    return found


def _usage_summary(done: dict[str, dict], keys: list[str]) -> str:
    counts: dict[str, int] = {}
    for key in keys:
        label = done.get(key, {}).get("llm") or "?"
        counts[label] = counts.get(label, 0) + 1
    return ", ".join(f"{label} x{n}" for label, n in counts.items())


def candidates_per_window(max_clips: int, window_count: int) -> int:
    """Keep a small, diverse pool without paying to generate dozens of losers."""
    return max(1, min(4, (max_clips + max(1, window_count) - 1) // max(1, window_count) + 1))


def detect(transcript: Transcript, peaks: list[Peak], llm: JsonLLM, max_clips: int,
           min_dur: float, max_dur: float, cache: Path,
           excluded: tuple[tuple[float, float], ...] = (),
           preference: SpeakerPreference | None = None,
           window_seconds: float = 900.0,
           content_type: str = "podcast",
           campaign_context: str = "", gameplay_criteria: gameplay.GameplayCriteria | None = None) -> list[Highlight]:
    spans = windows(transcript.duration, size=window_seconds)
    criteria = (gameplay_criteria or gameplay.GameplayCriteria()) if content_type == "gameplay" else None
    units = gameplay.speech_units(transcript.words) if criteria else []
    clusters = gameplay.reaction_clusters(units, peaks, min_dur, max_dur) if criteria else []
    report_path = cache.with_suffix(".candidates.json")
    per_chunk = candidates_per_window(max_clips, len(spans))
    system = SYSTEM_PROMPT.format(min_dur=int(min_dur), max_dur=int(max_dur),
                                  per_chunk=per_chunk, language=transcript.language)
    system += CONTENT_GUIDANCE.get(content_type, "")
    if criteria:
        system = gameplay.SYSTEM
    if campaign_context:
        system += "\nCampaign requirements:\n" + campaign_context + "\nChoose only moments that can honestly fit these requirements.\n"
    if preference:
        system += preference.prompt()
    params = {"min_dur": min_dur, "max_dur": max_dur, "per_chunk": per_chunk,
              "max_clips": max_clips, "excluded": [list(span) for span in excluded],
              "window_seconds": window_seconds,
              "content_type": content_type,
              **({"gameplay_prompt": 4, "criteria": asdict(criteria),
                  "transcript": hashlib.sha256(repr(transcript).encode()).hexdigest()} if criteria else {}),
              "campaign_fingerprint": hashlib.sha256(campaign_context.encode()).hexdigest() if campaign_context else "",
              "signals_fingerprint": hashlib.sha256(json.dumps([asdict(p) for p in peaks], sort_keys=True).encode()).hexdigest()}
    if preference:
        params["speaker_preference"] = preference.fingerprint
    progress = progress_path(cache)
    done = _load_progress(progress, params)
    if cache.is_file() and progress.is_file() and (not criteria or report_path.is_file()):
        saved = json.loads(progress.read_text(encoding="utf-8"))
        if saved.get("params") == params:
            return [Highlight(**h) for h in json.loads(cache.read_text(encoding="utf-8"))]
    keys = []
    pending = []
    batches = {}
    report = {}
    for index, (start, end) in enumerate(spans, 1):
        if preference and preference.mode == "only" and not any(
                t.speaker.casefold() == preference.speaker.casefold() and t.start < end and t.end > start
                for t in preference.turns):
            continue
        key = _window_key(start, end)
        if criteria:
            boundary = spans[index][0] if index < len(spans) else end
            pool = [c for c in clusters if start <= c["start"] < boundary]
            batch = sorted(pool, key=lambda c: c["reaction_strength"], reverse=True)[:max(4, per_chunk)]
            for cluster in pool:
                report[cluster["id"]] = cluster | {"status": "below_reaction_shortlist"}
            if not batch:
                continue
            batches[key] = batch
        else:
            words = transcript.words_between(start, end)
            if len(words) < 30:
                continue
        keys.append(key)
        if key in done:
            print(f"[detect] chunk {index}/{len(spans)} already done by {done[key].get('llm') or '?'}", flush=True)
            continue
        print(f"[detect] chunk {index}/{len(spans)} ({start:.0f}s-{end:.0f}s)", flush=True)
        if criteria:
            user = gameplay.format_batch(batch, units, preference, min_dur, max_dur)
        else:
            user = "\n\n".join(part for part in (
                format_hints(peaks, start, end), "Transcript:\n" + format_lines(words, preference=preference)) if part)
        already_used = [(a, b) for a, b in excluded if a < end and b > start]
        if already_used:
            user += ("\n\nThese intervals already became clips. Choose DIFFERENT moments and avoid overlap: "
                     + json.dumps(already_used))
        try:
            done[key] = _ask_window(llm, system, user, index, gameplay_batch=bool(criteria))
        except CodexResponseRequired as exc:
            pending.extend(exc.requests)
            continue
        except ProvidersExhausted as exc:
            raise ProvidersExhausted(
                f"all LLMs failed at chunk {index}/{len(spans)}; progress kept in {progress}, "
                f"run the same command again to resume. Errors: {exc}") from exc
        _save_progress(progress, params, done)
    if pending:
        raise CodexResponseRequired(list(dict.fromkeys(pending)), "selection")
    if keys:
        print(f"[detect] chunks by LLM: {_usage_summary(done, keys)}", flush=True)
    previous = [Highlight(a, b, 0, "", "", "") for a, b in excluded]
    if criteria:
        candidates = []
        for key, batch in batches.items():
            verdicts = {item["id"]: item for item in done[key].get("reviews", [])}
            for cluster in batch:
                identifier = cluster["id"]
                item = verdicts.get(identifier)
                bounds, rejection = gameplay.validate(item, cluster, units, criteria, min_dur, max_dur, transcript.duration)
                entry = report[identifier] = cluster | {"verdict": item, "status": rejection}
                if item:
                    entry["final_score"] = criteria.score(item)
                if not bounds:
                    continue
                start, end = bounds
                score = entry["final_score"]
                entry["clip_bounds"] = bounds
                clip = Highlight(start, end, round(score), item["title"][:70], item["hook"], item["reason"],
                                 done[key]["llm"], selection_score=score)
                if preference:
                    clip.speaker_focus = preference.evaluate(transcript.words, start, end)
                    if not clip.speaker_focus["eligible"]:
                        entry["status"] = "Speaker restriction"
                        continue
                    clip.selection_score = round((1 - preference.weight) * score
                                                 + preference.weight * 100 * clip.speaker_focus["preferred_speech_share"], 2)
                if any(overlap_ratio(clip, old) >= 0.3 for old in previous):
                    entry["status"] = "Overlaps an existing clip"
                    continue
                entry["status"] = "Outside ranking limit or overlaps a higher-ranked candidate"
                candidates.append((identifier, clip))
        chosen = dedupe([clip for _, clip in candidates])[:max_clips]
        for identifier, clip in candidates:
            if any(clip is selected for selected in chosen):
                report[identifier]["status"] = "selected"
        report_path.write_text(json.dumps({"status": "selected" if chosen else "no_qualified_candidates",
                                           "candidates": list(report.values())}, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        print(f"[gameplay] {len(chosen)} qualified clips; report: {report_path}", flush=True)
    else:
        candidates = _snap_all(done, keys, transcript, min_dur, max_dur, preference)
        candidates = [h for h in candidates if all(overlap_ratio(h, old) < 0.3 for old in previous)]
        chosen = dedupe(candidates)[:max_clips]
    _save_progress(progress, params, done)
    cache.write_text(json.dumps([asdict(h) for h in chosen], ensure_ascii=False, indent=2), encoding="utf-8")
    return chosen
