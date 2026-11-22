import json

import numpy as np

from clipper import detect, gameplay_audio
from clipper.gameplay import _sample_times
from clipper.gameplay_selection import GameplayCriteria, reaction_clusters
from clipper.transcribe import Transcript, Word


def test_pitch_and_intensity_distinguish_corroboration_from_game_noise():
    t = np.arange(gameplay_audio.RATE) / gameplay_audio.RATE
    rng = np.random.default_rng(4)
    chunks = [np.sin(2 * np.pi * 125 * t) * 1000 for _ in range(24)]
    chunks[12] = np.sin(2 * np.pi * 250 * t) * 5500
    chunks[21] = rng.normal(0, 8000, len(t))
    words = [Word("Não acredito!" if i in (3, 12) else "fala.", i + .05, i + .95)
             for i in range(20)]
    events = gameplay_audio.analyze(chunks, words, 24)
    by_time = {event.start: event for event in events}
    assert by_time[12].features["pitch_shift_semitones"] > 10
    assert by_time[12].features["onset_z"] > 3
    assert by_time[21].event_type == "intensity_change"
    assert "pitch_hz" not in by_time[21].features
    assert not gameplay_audio.analyze(chunks[:3], [], 3)
    assert reaction_clusters([], [], 5, 20, events) == []


def test_fusion_prioritizes_the_scene_after_editorial_approval(tmp_path):
    words = [
        Word("Primeira tentativa.", 1, 2), Word("Vou subir.", 9, 10),
        Word("Meu Deus!", 12, 13), Word("Cheguei no topo.", 15, 17),
        Word("Segunda tentativa.", 30, 31), Word("Vou saltar.", 34, 35),
        Word("Meu Deus!", 38, 39), Word("Cheguei do outro lado.", 40, 42),
        Word("Eita!", 61, 70),
    ]
    event = gameplay_audio.AudioEvent(38, 39, 38.2, "pitch_change", 8, .92,
                                     {"pitch_z": 7, "pitch_confidence": .92, "onset_z": 6})
    tx = Transcript("pt", 90, words)
    calls = []

    def verdict(system, user, schema):
        data = json.loads(user)
        calls.append(data)
        assert "missing causes or endings" in system
        return {"reviews": [
            dict(id=i, accepted=True, first=a, development=b, payoff=c, last=d,
                 title="Corte", hook="Gancho", reason="Motivo", **dict.fromkeys(GameplayCriteria().weights, 85))
            for i, (a, b, c, d) in enumerate([(0, 1, 2, 3), (4, 5, 6, 7), (8, 8, 8, 8)])
        ]}

    cache = tmp_path / "highlights.json"
    def select(events):
        return detect.detect(tx, [], verdict, 2, 8, 20, cache, content_type="gameplay", gameplay_audio=events)

    assert [clip.start for clip in select(None)] == [1, 30]
    assert [clip.start for clip in select([event])] == [30, 1]
    assert [clip.ranking_score for clip in select([event])] == [88, 85]
    assert len(calls) == 2
    second = next(cluster for cluster in calls[1]["clusters"] if cluster["id"] == 1)
    assert second["lead_in_start"] == 30
    assert second["acoustic_support"] == 1
    assert any(unit[1] == 30 for unit in calls[1]["speech"])
    report = json.loads(cache.with_suffix(".candidates.json").read_text(encoding="utf-8"))
    assert next(row for row in report["candidates"] if row["id"] == 2)["status"] == "Cut lacks a setup before the payoff"


def test_acoustic_cache_tracks_transcript_and_version(tmp_path, monkeypatch):
    video = tmp_path / "video.mp4"
    video.touch()
    cache = tmp_path / "gameplay_audio_events.json"
    calls = []

    def decode(*args, **kwargs):
        calls.append(args)
        return iter([np.zeros(gameplay_audio.RATE)])

    monkeypatch.setattr(gameplay_audio, "iter_audio_seconds", decode)
    tx = Transcript("pt", 1)
    assert gameplay_audio.collect(video, tx, cache) == []
    assert gameplay_audio.collect(video, tx, cache) == []
    assert len(calls) == 1
    tx.words.append(Word("fala", .1, .7))
    assert gameplay_audio.collect(video, tx, cache) == []
    monkeypatch.setattr(gameplay_audio, "VERSION", 2)
    assert gameplay_audio.collect(video, tx, cache) == []
    assert len(calls) == 3


def test_missing_audio_is_distinct_from_a_valid_empty_analysis(tmp_path, monkeypatch):
    video = tmp_path / "video.mp4"
    video.touch()

    def fail(*args, **kwargs):
        raise gameplay_audio.FFmpegError("no audio stream")

    monkeypatch.setattr(gameplay_audio, "iter_audio_seconds", fail)
    cache = tmp_path / "gameplay_audio_events.json"
    assert gameplay_audio.collect(video, Transcript("pt", 10), cache) is None
    assert not cache.exists()


def test_visual_sheet_samples_the_cause_and_after_reaction():
    times = _sample_times(1, 24, [], reaction=12)
    assert len(times) == 8
    assert {3, 4, 5.8, 10, 12, 15} <= set(times)
    assert all(1 <= t < 24 for t in times)


def test_audio_describes_changes_in_speech_rate_and_pauses():
    t = np.arange(gameplay_audio.RATE) / gameplay_audio.RATE
    tone = np.sin(2 * np.pi * 125 * t) * 1000
    chunks = [tone.copy() if not 14 <= i < 17 else np.zeros_like(tone) for i in range(24)]
    words = [Word("fala", i + .1, i + .8) for i in range(24) if not 10 <= i < 12 and not 14 <= i < 17]
    words.extend(Word("rapido", i + .2 * j, i + .2 * j + .15) for i in (10, 11) for j in range(5))

    events = gameplay_audio.analyze(chunks, words, 24)

    assert any(event.features["pace_z"] > 2.5 and 9 <= event.start <= 12 for event in events)
    assert any(event.features["pause_seconds"] > 3 for event in events)
