"""Review a bounded gameplay shortlist using local contact sheets."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np

from clipper.codex_exchange import CodexExchange
from clipper.detect import Highlight, dedupe, overlap_ratio
from clipper.signals import Peak
from clipper.transcribe import Transcript

SYSTEM = """Review gameplay candidates using their transcript excerpts and labeled frame grids.
Pick moments with a clear visual or spoken hook, readable action and a payoff that works on its own.
Action, surprise or a funny outcome can carry a clip without dialogue. A loud peak alone is weak.
Frames are samples, so avoid claiming anything they and the transcript do not establish.
Return only selected indices with a viral potential score from 0 to 100, a short title, an honest hook
(spoken or visual), and a concise reason. Do not repeat equivalent moments."""

FIELDS = {"index": {"type": "integer"}, "score": {"type": "integer", "minimum": 0, "maximum": 100},
          "title": {"type": "string", "minLength": 1, "maxLength": 70},
          "hook": {"type": "string", "minLength": 1}, "reason": {"type": "string", "minLength": 1}}
SCHEMA = {"type": "object", "required": ["clips"], "additionalProperties": False, "properties": {
    "clips": {"type": "array", "items": {"type": "object", "properties": FIELDS,
                                       "required": list(FIELDS), "additionalProperties": False}},
}}


def _sheet(capture, path: Path, start: float, end: float) -> str | None:
    if not path.is_file():
        frames = []
        span = end - start
        for offset in (min(0.3, span / 8), min(2, span / 3), span / 2, span - min(0.5, span / 8)):
            time = start + offset
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
            cv2.rectangle(tile, (0, 155), (100, 179), (0, 0, 0), -1)
            cv2.putText(tile, f"{time:.1f}s", (6, 173), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            frames.append(tile)
        grid = np.vstack((np.hstack(frames[:2]), np.hstack(frames[2:])))
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


def _peak_clips(peaks: list[Peak], duration: float, min_dur: float, max_dur: float,
                existing: list[Highlight]) -> list[Highlight]:
    length = min(duration, max_dur, max(min_dur, 28.0))
    if length < min_dur:
        return []
    chosen = []
    for peak in sorted((p for p in peaks if p.kind == "audio" and 0 <= p.time < duration),
                       key=lambda p: p.strength, reverse=True):
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
           campaign_context: str = "") -> list[Highlight]:
    capture = cv2.VideoCapture(str(video))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not capture.isOpened() or fps <= 0 or frames <= 0:
            print("[gameplay] video frames unavailable; using transcript selection", flush=True)
            return highlights[:limit]
        duration = frames / fps
        candidates = [clip for clip in highlights if clip.end <= duration][:12]
        if speaker_weight is None:
            occupied = candidates + [Highlight(a, b, 0, "", "", "") for a, b in excluded]
            candidates.extend(_peak_clips(peaks, duration, minimum, maximum, occupied))
        folder = exchange.directory / "frames"
        folder.mkdir(parents=True, exist_ok=True)
        stat = video.stat()
        images = []
        details = []
        for index, clip in enumerate(candidates):
            key = f"{video.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{clip.start}:{clip.end}"
            path = folder / (hashlib.sha256(key.encode()).hexdigest() + ".jpg")
            digest = _sheet(capture, path, clip.start, clip.end)
            if digest:
                images.append({"index": index, "file": f"frames/{path.name}", "sha256": digest})
            details.append({"index": index, "start": clip.start, "end": clip.end, "title": clip.title,
                            "has_frames": bool(digest), "transcript": " ".join(
                                word.text for word in transcript.words_between(clip.start, clip.end))[:1000]})
    finally:
        capture.release()
    if not images:
        print("[gameplay] no preview frames decoded; using transcript selection", flush=True)
        return [clip for clip in candidates if clip.title][:limit]
    system = SYSTEM + f" Select at most {limit} clips. Write in language {transcript.language}."
    if campaign_context:
        system += "\nCampaign requirements:\n" + campaign_context
    print(f"[gameplay] visual review: {len(images)} frame grids", flush=True)
    responses = exchange(system, json.dumps(details, ensure_ascii=False), SCHEMA, images=images)["clips"]
    visible = {image["index"] for image in images}
    selected = {}
    for item in responses:
        index = item["index"]
        if index in selected or not 0 <= index < len(candidates):
            continue
        clip = candidates[index]
        if not clip.title and index not in visible:
            continue
        score = item["score"]
        ranking = None
        if speaker_weight is not None and clip.speaker_focus:
            share = clip.speaker_focus["preferred_speech_share"]
            ranking = round((1 - speaker_weight) * score + speaker_weight * 100 * share, 2)
        selected[index] = replace(clip, score=score, title=item["title"], hook=item["hook"],
                                  reason=item["reason"], llm=f"codex:{exchange.model}", selection_score=ranking)
    return dedupe(list(selected.values()))[:limit]
