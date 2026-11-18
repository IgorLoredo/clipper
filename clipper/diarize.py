"""Local anonymous speaker diarization and reference-based name mapping."""

import hashlib
import json
import math
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np

from clipper import ffmpeg


def non_overlapping_turns(rows: list[dict]) -> list[dict]:
    events = defaultdict(list)
    for row in rows:
        start, end = float(row["start"]), float(row["end"])
        if not all(math.isfinite(x) for x in (start, end)) or start < 0 or end <= start:
            continue
        speaker = str(row["speaker"])
        events[start].append((speaker, 1))
        events[end].append((speaker, -1))
    active = defaultdict(int)
    times = sorted(events)
    result = []
    for index, start in enumerate(times[:-1]):
        for speaker, change in events[start]:
            active[speaker] += change
        speakers = [speaker for speaker, count in active.items() if count > 0]
        end = times[index + 1]
        if len(speakers) != 1:
            continue  # overlapping/uncertain speech remains unassigned
        speaker = speakers[0]
        if result and result[-1]["speaker"] == speaker and abs(result[-1]["end"] - start) < 1e-6:
            result[-1]["end"] = end
        else:
            result.append({"start": start, "end": end, "speaker": speaker})
    return result


def map_anchors(turns: list[dict], anchors: list[str]) -> tuple[list[dict], list[dict]]:
    names, evidence = {}, []
    for spec in anchors:
        try:
            name, span = spec.rsplit("=", 1)
            start, end = map(float, span.split(":"))
            if not name.strip() or not 0 <= start < end or not all(math.isfinite(x) for x in (start, end)):
                raise ValueError
        except ValueError as exc:
            raise ValueError("anchor must be NAME=START:END in seconds") from exc
        durations = defaultdict(float)
        for turn in turns:
            overlap = max(0.0, min(end, turn["end"]) - max(start, turn["start"]))
            durations[turn["speaker"]] += overlap
        ranked = sorted(durations.items(), key=lambda item: item[1], reverse=True)
        if not ranked or ranked[0][1] / (end - start) < 0.6:
            raise ValueError(f"speaker anchor {spec!r} has insufficient unambiguous speech")
        speaker, matched = ranked[0]
        if len(ranked) > 1 and (matched - ranked[1][1]) / (end - start) < 0.25:
            raise ValueError(f"speaker anchor {spec!r} is ambiguous")
        name = name.strip()
        if speaker in names and names[speaker].casefold() != name.casefold():
            raise ValueError("different names were assigned to the same anonymous speaker")
        if any(old_name.casefold() == name.casefold() and old_id != speaker for old_id, old_name in names.items()):
            raise ValueError("references for the same participant disagree; review the anonymous speaker labels")
        names[speaker] = name
        evidence.append({"name": name, "speaker_id": speaker, "start": start, "end": end,
                         "reference_coverage": round(matched / (end - start), 4)})
    return [{**turn, "speaker": names.get(turn["speaker"], turn["speaker"])} for turn in turns], evidence


def _parse_anchors(anchors: list[str]) -> list[tuple[str, float, float]]:
    parsed = []
    for spec in anchors:
        try:
            name, span = spec.rsplit("=", 1)
            start, end = map(float, span.split(":"))
            if not name.strip() or not 0 <= start < end or not all(math.isfinite(x) for x in (start, end)):
                raise ValueError
        except ValueError as exc:
            raise ValueError("anchor must be NAME=START:END in seconds") from exc
        parsed.append((name.strip(), start, end))
    if len({name.casefold() for name, _, _ in parsed}) != len(parsed):
        # Multiple examples for a name are useful; keep them, but names must be non-empty.
        return parsed
    return parsed


def _voice_embedding(extractor, samples: np.ndarray) -> np.ndarray:
    stream = extractor.create_stream()
    stream.accept_waveform(16000, samples)
    stream.input_finished()
    while not extractor.is_ready(stream):
        pass
    vector = np.asarray(extractor.compute(stream), dtype=np.float32)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def _span_audio(samples: np.ndarray, start: float, end: float) -> np.ndarray:
    start_index = max(0, int(start * 16000))
    end_index = min(len(samples), int(end * 16000))
    return samples[start_index:end_index]


def _prototypes(extractor, samples: np.ndarray, anchors: list[tuple[str, float, float]]) -> dict[str, np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for name, start, end in anchors:
        # One embedding every ten seconds makes a long reference resilient to a brief interruption.
        cursor = start
        while cursor < end:
            piece = _span_audio(samples, cursor, min(end, cursor + 10))
            if len(piece) >= 3 * 16000:
                grouped[name].append(_voice_embedding(extractor, piece))
            cursor += 10
    if not grouped:
        raise ValueError("at least one reference interval is required")
    result = {}
    for name, vectors in grouped.items():
        vector = np.mean(vectors, axis=0)
        result[name] = vector / max(float(np.linalg.norm(vector)), 1e-12)
    return result


def _classify_chunk_turns(turns: list[dict], samples: np.ndarray, offset: float, extractor,
                          prototypes: dict[str, np.ndarray]) -> tuple[list[dict], list[dict]]:
    """Label only clearly matching local speaker groups; preserve uncertainty as anonymous."""
    clean = non_overlapping_turns(turns)
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for turn in clean:
        if turn["end"] - turn["start"] >= 3:
            by_speaker[turn["speaker"]].append(turn)
    labels, evidence = {}, []
    for local_id, local_turns in by_speaker.items():
        vectors = []
        # Longest clean speech portions give a stable identity signal without treating overlap as speech.
        for turn in sorted(local_turns, key=lambda item: item["end"] - item["start"], reverse=True)[:3]:
            start = offset + turn["start"]
            end = min(offset + turn["end"], start + 10)
            piece = _span_audio(samples, start, end)
            if len(piece) >= 3 * 16000:
                vectors.append(_voice_embedding(extractor, piece))
        if not vectors:
            continue
        vector = np.mean(vectors, axis=0)
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        scores = sorted(((float(vector @ prototype), name) for name, prototype in prototypes.items()), reverse=True)
        score, name = scores[0]
        second = scores[1][0] if len(scores) > 1 else -1.0
        # Voiceprints are approximate. A conservative score and margin avoid inventing a name.
        if score >= 0.86 and score - second >= 0.035:
            labels[local_id] = name
            evidence.append({"speaker_id": local_id, "name": name, "similarity": round(score, 4),
                             "margin": round(score - second, 4)})
    result = []
    for turn in clean:
        result.append({"start": offset + turn["start"], "end": offset + turn["end"],
                       "speaker": labels.get(turn["speaker"], f"anonymous_{turn['speaker']}")})
    return result, evidence


def run(workdir: Path, segmentation_model: Path, embedding_model: Path, num_speakers: int = 4,
        threshold: float = 0.25, window_shift: float = 0.5, anchors: list[str] | None = None,
        output: Path | None = None, chunk_seconds: int = 300) -> Path:
    try:
        import sherpa_onnx
    except ImportError as exc:
        raise ValueError('install local diarization with pip install -e ".[diarization]"') from exc
    from clipper.transcribe import Transcript

    for model in (segmentation_model, embedding_model):
        if not model.is_file():
            raise ValueError(f"diarization model not found: {model}")
    duration = Transcript.load(workdir / "transcript.json").duration
    metadata = json.loads((workdir / "source.json").read_text(encoding="utf-8"))
    video = Path(metadata["video"]).resolve()
    parsed_anchors = _parse_anchors(anchors or [])
    if not parsed_anchors:
        raise ValueError("provide reference voice intervals with --anchor NAME=START:END")
    if chunk_seconds < 60:
        raise ValueError("chunk_seconds must be at least 60")
    signature = {
        "video": str(video), "size": video.stat().st_size, "mtime": video.stat().st_mtime_ns,
        "segmentation": str(segmentation_model.resolve()), "embedding": str(embedding_model.resolve()),
        "num_speakers": num_speakers, "threshold": threshold, "window_shift": window_shift,
        "chunk_seconds": chunk_seconds, "anchors": parsed_anchors,
    }
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:12]
    folder = workdir / "diarization"
    folder.mkdir(parents=True, exist_ok=True)
    cache = folder / f"anonymous.{digest}.json"
    cached = json.loads(cache.read_text(encoding="utf-8")) if cache.is_file() else {}
    if cached.get("complete"):
        turns, identity_evidence = cached["turns"], cached["identity_evidence"]
        print("[diarize] reusing cached locally classified speaker groups", flush=True)
    else:
        turns = cached.get("turns", [])
        identity_evidence = cached.get("identity_evidence", [])
        completed_chunks = int(cached.get("completed_chunks", 0))
        if completed_chunks:
            print(f"[diarize] resuming after chunk {completed_chunks}", flush=True)
        audio = folder / f"audio.{video.stat().st_size}.wav"
        if not audio.is_file():
            print("[diarize] extracting 16 kHz mono audio", flush=True)
            temporary = audio.with_suffix(".part.wav")
            ffmpeg.run(["-i", str(video), "-vn", "-ar", "16000", "-ac", "1", str(temporary.resolve())])
            temporary.replace(audio)
        with wave.open(str(audio), "rb") as wav:
            if wav.getframerate() != 16000 or wav.getnchannels() != 1 or wav.getsampwidth() != 2:
                raise ValueError("diarization audio must be 16 kHz, mono, PCM16")
            samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(segmentation_model), window_shift_ratio=window_shift), num_threads=4),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(embedding_model), num_threads=4),
            clustering=sherpa_onnx.FastClusteringConfig(num_clusters=num_speakers, threshold=threshold),
            min_duration_on=0.3, min_duration_off=0.5,
        )
        if not config.validate():
            raise ValueError("invalid local diarization configuration")
        extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(embedding_model), num_threads=4))
        prototypes = _prototypes(extractor, samples, parsed_anchors)
        turns, identity_evidence = [], []
        if completed_chunks:
            turns = cached["turns"]
            identity_evidence = cached["identity_evidence"]
        chunks = max(1, math.ceil(duration / chunk_seconds))
        for index, offset in enumerate(range(0, math.ceil(duration), chunk_seconds), start=1):
            if index <= completed_chunks:
                continue
            end = min(duration, offset + chunk_seconds)
            engine = sherpa_onnx.OfflineSpeakerDiarization(config)
            result = engine.process(_span_audio(samples, offset, end)).sort_by_start_time()
            raw = [{"start": max(0.0, float(row.start)), "end": min(end - offset, float(row.end)),
                    "speaker": f"speaker_{row.speaker:02d}"} for row in result]
            classified, evidence = _classify_chunk_turns(raw, samples, offset, extractor, prototypes)
            turns.extend(classified)
            identity_evidence.extend([{"chunk_start": offset, **item} for item in evidence])
            temporary = cache.with_suffix(".tmp")
            temporary.write_text(json.dumps({"signature": signature, "turns": turns,
                                             "identity_evidence": identity_evidence,
                                             "completed_chunks": index, "complete": False}), encoding="utf-8")
            temporary.replace(cache)
            print(f"[diarize] chunk {index}/{chunks}", flush=True)
        temporary = cache.with_suffix(".tmp")
        temporary.write_text(json.dumps({"signature": signature, "turns": turns,
                                         "identity_evidence": identity_evidence,
                                         "completed_chunks": chunks, "complete": True}), encoding="utf-8")
        temporary.replace(cache)
    path = output or workdir / "speaker-turns.json"
    if path.exists():
        base, index = path, 2
        while path.exists():
            path = base.with_name(f"{base.stem}-{index:02d}{base.suffix}")
            index += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "source": "local_diarization_with_reference_voice_matching", "version": 2,
        "anonymous_cache": str(cache), "references": [{"name": n, "start": s, "end": e} for n, s, e in parsed_anchors],
        "identity_evidence": identity_evidence,
        "limitations": "Names come from local voice-reference matching. Overlap and uncertain matches remain anonymous; review labels before publication.",
        "turns": turns,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[diarize] {len(turns)} non-overlapping turns written to {path}", flush=True)
    return path
