"""Review a bounded gameplay shortlist using local contact sheets."""

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np

from clipper.codex_exchange import CodexExchange
from clipper.detect import QUERY_GUIDANCE, Highlight, dedupe, format_lines, overlap_ratio
from clipper.gameplay_audio import AudioEvent
from clipper.gameplay_selection import GameplayCriteria, HOOK_GUIDANCE
from clipper.signals import Peak, find_peaks
from clipper.transcribe import Transcript

SYSTEM = """Review gameplay candidates using their transcript excerpts and labeled frame grids.
Pick moments with a clear visual or spoken hook, readable action and a payoff that works on its own.
Action, surprise or a funny outcome can carry a clip without dialogue. A loud peak alone is weak.
Frames are samples, so avoid claiming anything they and the transcript do not establish.
When a reaction timestamp is available, check what happened before it, whether the action explains
the reaction, and whether the ending resolves the setup. Reject clips that require unseen context.
Use the frames to challenge the transcript interpretation. A visual or acoustic peak is just a
location hint. Volume, pitch and frame changes do not establish a player's emotion, a win or a failure.
For candidates without dialogue, verify readable action, a significant payoff and standalone clarity.
Use timestamped dialogue, acoustic cues and nearby frames together. For humor, locate what
actually makes the scene funny, including quiet delivery and purely visual reversals. Treat
laughter as a clue to the preceding event and consider whether sounds come from the game.
For an unranked candidate, select it only when the frames independently establish the key
event and provide enough evidence for the setup and payoff.
Examine the first five seconds as an opening. Look for understandable action or dialogue
and a reason to keep watching before the payoff. A strong payoff alone cannot rescue a
confusing opening. Base this judgment on the provided source evidence, without predicting retention.
Return only selected indices with a viral potential score from 0 to 100, a short title, an honest hook
(spoken or visual), and a concise reason. Do not repeat equivalent moments.
Compare the existing hook with the transcript and frame sequence. Return the existing hook
unchanged when the frames do not substantiate a better interpretation. If they do, return
the improved hook grounded in the action and reaction. Preserve the original surprise.""" + "\n\n" + HOOK_GUIDANCE

FIELDS = {"index": {"type": "integer"}, "score": {"type": "integer", "minimum": 0, "maximum": 100},
          "title": {"type": "string", "minLength": 1, "maxLength": 70},
          "hook": {"type": "string", "minLength": 1}, "reason": {"type": "string", "minLength": 1}}
SCHEMA = {"type": "object", "required": ["clips"], "additionalProperties": False, "properties": {
    "clips": {"type": "array", "items": {"type": "object", "properties": FIELDS,
                                       "required": list(FIELDS), "additionalProperties": False}},
}}


def _sample_times(start: float, end: float, peaks: list[Peak], reaction: float | None = None) -> list[float]:
    span = end - start
    offsets = [min(0.3, span / 10), min(2, span * 0.2), min(4.8, span * 0.4),
               span * 0.4, span * 0.5, span * 0.6, span * 0.8, span - min(0.5, span / 10)]
    inside = [peak for peak in peaks if start <= peak.time < end]
    if reaction is not None and start <= reaction < end:
        at = reaction - start
        offsets[3:7] = [max(0, at - 8), max(0, at - 2), at, min(span - .05, at + 3)]
    elif inside:
        strongest = max(inside, key=lambda peak: peak.strength)
        offsets[5] = strongest.time - start
    return sorted(start + max(0.0, min(span, offset)) for offset in offsets)


def _sheet(capture, path: Path, times: list[float]) -> str | None:
    if not path.is_file():
        frames = []
        for time in times:
            capture.set(cv2.CAP_PROP_POS_MSEC, time * 1000)
            ok, frame = capture.read()
            if not ok:
                return None
            height, width = frame.shape[:2]
            scale = min(320 / width, 180 / height)
            resized = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))),
                                 interpolation=cv2.INTER_AREA)
            tile = np.zeros((180, 320, 3), dtype=np.uint8)
            y, x = (180 - resized.shape[0]) // 2, (320 - resized.shape[1]) // 2
            tile[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
            cv2.rectangle(tile, (0, 151), (319, 179), (0, 0, 0), -1)
            minutes, seconds = divmod(time, 60)
            cv2.putText(tile, f"{int(minutes):02d}:{seconds:04.1f}", (8, 172),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
            frames.append(tile)
        grid = np.vstack((np.hstack(frames[:4]), np.hstack(frames[4:])))
        ok, jpeg = cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return None
        with NamedTemporaryFile(dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            file.write(jpeg.tobytes())
        try:
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _visual_peaks(capture, duration: float) -> list[Peak]:
    """Find scene/motion changes locally, capped to keep long videos inexpensive."""
    step = max(1.0, duration / 1200.0)
    scores, previous_gray, previous_hist = [], None, None
    time = 0.0
    while time < duration:
        capture.set(cv2.CAP_PROP_POS_MSEC, time * 1000)
        ok, frame = capture.read()
        if not ok:
            scores.append(0.0)
            time += step
            continue
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([small], [0, 1], None, [16, 16], [0, 256, 0, 256])
        cv2.normalize(hist, hist)
        if previous_gray is None:
            score = 0.0
        else:
            motion = float(np.mean(cv2.absdiff(gray, previous_gray))) / 255.0
            scene = float(cv2.compareHist(previous_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
            score = motion + 0.65 * scene
        scores.append(score)
        previous_gray, previous_hist = gray, hist
        time += step
    if len(scores) < 3:
        return []
    return find_peaks(np.asarray(scores), step, "visual", threshold=2.0, min_gap=18.0, limit=8)


def _cached_visual_peaks(capture, duration: float, video: Path, cache: Path) -> list[Peak]:
    stat = video.stat()
    key = [1, str(video.resolve()), stat.st_size, stat.st_mtime_ns, duration]
    if cache.is_file():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if data["key"] == key:
                return [Peak(**peak) for peak in data["peaks"]]
        except (OSError, ValueError, KeyError, TypeError):
            pass
    result = _visual_peaks(capture, duration)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=cache.parent, delete=False) as file:
        temporary = Path(file.name)
        json.dump({"key": key, "peaks": [asdict(peak) for peak in result]}, file)
    try:
        temporary.replace(cache)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def _peak_clips(peaks: list[Peak], duration: float, min_dur: float, max_dur: float,
                existing: list[Highlight]) -> list[Highlight]:
    length = min(duration, max_dur, max(min_dur, 28.0))
    if length < min_dur:
        return []
    chosen = []
    for peak in sorted((p for p in peaks if p.kind in {"audio", "visual"} and 0 <= p.time < duration),
                       key=lambda p: (p.kind == "visual", p.strength), reverse=True):
        start = min(max(0.0, peak.time - length * 0.45), duration - length)
        clip = Highlight(start, start + length, 0, "", "", "")
        if all(overlap_ratio(clip, other) < 0.3 for other in existing + chosen):
            chosen.append(clip)
        if len(chosen) == 4:
            break
    return chosen


def review(video: Path, transcript: Transcript, highlights: list[Highlight], peaks: list[Peak],
           excluded: tuple[tuple[float, float], ...], minimum: float, maximum: float, limit: int,
           exchange: CodexExchange, *, speaker_weight: float | None,
           campaign_context: str = "", editorial: list[dict] | None = None,
           audio_events: list[AudioEvent] | None = None, criteria: GameplayCriteria | None = None,
           report_path: Path | None = None, query: str = "") -> list[Highlight]:
    capture = cv2.VideoCapture(str(video))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not capture.isOpened() or fps <= 0 or frames <= 0:
            print("[gameplay] video frames unavailable; using transcript selection", flush=True)
            return highlights[:limit]
        duration = frames / fps
        folder = exchange.directory / "frames"
        folder.mkdir(parents=True, exist_ok=True)
        review_peaks = peaks + _cached_visual_peaks(capture, duration, video, folder / "visual_peaks.json")
        evidence = {tuple(entry["clip_bounds"]): entry for entry in editorial or [] if "clip_bounds" in entry}
        candidates = [clip for clip in highlights if clip.end <= duration][:12]
        if speaker_weight is None:
            occupied = candidates + [Highlight(a, b, 0, "", "", "") for a, b in excluded]
            rescue_limit = len(candidates) + 4
            for entry in sorted(editorial or [], key=lambda item: item.get("final_score", 0), reverse=True):
                if "visual_review_bounds" not in entry:
                    continue
                start, end = entry["visual_review_bounds"]
                clip = Highlight(start, end, 0, "", "", "")
                if end <= duration and all(overlap_ratio(clip, other) < 0.3 for other in occupied):
                    candidates.append(clip)
                    occupied.append(clip)
                    evidence[(start, end)] = entry
                    if len(candidates) >= rescue_limit:
                        break
            candidates.extend(_peak_clips(review_peaks, duration, minimum, maximum, occupied))
        stat = video.stat()
        images = []
        details = []
        for index, clip in enumerate(candidates):
            entry = evidence.get((clip.start, clip.end), {})
            reaction = next((t for t in entry.get("reaction_times", []) if clip.start <= t < clip.end), None)
            times = _sample_times(clip.start, clip.end, review_peaks, reaction)
            key = f"{video.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:eight-frames:{times}"
            path = folder / (hashlib.sha256(key.encode()).hexdigest() + ".jpg")
            digest = _sheet(capture, path, times)
            if digest:
                images.append({"index": index, "file": f"frames/{path.name}", "sha256": digest})
            clip_peaks = [peak for peak in review_peaks if clip.start <= peak.time <= clip.end]
            acoustic = sorted((event for event in audio_events or [] if clip.start <= event.peak_time < clip.end),
                              key=lambda event: event.strength, reverse=True)[:6]
            details.append({"index": index, "start": clip.start, "end": clip.end, "title": clip.title,
                            "hook": clip.hook,
                            "reaction_time": reaction, "frame_times": times,
                            "opening_end": min(clip.start + 5, clip.end),
                            "opening_transcript": format_lines(
                                transcript.words_between(clip.start, min(clip.start + 5, clip.end))),
                            "timed_transcript": format_lines(
                                transcript.words_between(clip.start, clip.end))[:2000],
                            "acoustic_events": [asdict(event) for event in acoustic],
                            "has_frames": bool(digest), "transcript": " ".join(
                                word.text for word in transcript.words_between(clip.start, clip.end))[:1000],
                            "signals": [{"time": peak.time, "kind": peak.kind,
                                         "strength": peak.strength} for peak in clip_peaks]})
    finally:
        capture.release()
    def save_report(status, verdicts=None, selected_ids=()):
        if report_path is None:
            return
        verdicts = verdicts or {}
        entries = [detail | {"status": "selected" if detail["index"] in selected_ids else status,
                             "verdict": verdicts.get(detail["index"])} for detail in details]
        report_path.write_text(json.dumps({"candidates": entries}, ensure_ascii=False, indent=2), encoding="utf-8")

    if not images:
        print("[gameplay] no preview frames decoded; using transcript selection", flush=True)
        save_report("frames_unavailable")
        return [clip for clip in candidates if clip.title][:limit]
    system = SYSTEM + f" Select at most {limit} clips. Write in language {transcript.language}."
    if highlights and all(clip.selection_score is not None for clip in highlights):
        system += " Use narrative ranking as a starting point. Exclude candidates whose frames undermine the story or whose opening needs missing context."
    if campaign_context:
        system += "\nCampaign requirements:\n" + campaign_context
    print(f"[gameplay] visual review: {len(images)} frame grids", flush=True)
    payload = details
    if query:
        system += QUERY_GUIDANCE
        payload = {"scene_query": query, "candidates": details}
    responses = exchange(system, json.dumps(payload, ensure_ascii=False), SCHEMA, images=images)["clips"]
    visible = {image["index"] for image in images}
    minimum_score = (criteria or GameplayCriteria()).minimum_score
    selected = {}
    for item in responses:
        index = item["index"]
        if index in selected or not 0 <= index < len(candidates):
            continue
        clip = candidates[index]
        if not clip.title and (index not in visible or item["score"] < minimum_score):
            continue
        score = item["score"]
        ranking = None
        if speaker_weight is not None and clip.speaker_focus:
            share = clip.speaker_focus["preferred_speech_share"]
            ranking = round((1 - speaker_weight) * score + speaker_weight * 100 * share, 2)
        if clip.selection_score is not None:
            hook = " ".join(item["hook"].split()) if index in visible else ""
            selected[index] = replace(clip, hook=hook) if hook else clip
        else:
            selected[index] = replace(clip, score=score, title=item["title"], hook=item["hook"],
                                      reason=item["reason"], llm=f"codex:{exchange.model}", selection_score=ranking)
    chosen = dedupe(list(selected.values()))[:limit]
    chosen_ids = {index for index, clip in selected.items() if any(clip is item for item in chosen)}
    save_report("rejected_or_below_ranking_limit", {item["index"]: item for item in responses}, chosen_ids)
    return chosen
