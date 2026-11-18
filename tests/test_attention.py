import json

import pytest

from clipper import attention, llm
from clipper.cli import main
from clipper.signals import Peak
from clipper.transcribe import Transcript


def band(start, end, value):
    return {"start_time": start, "end_time": end, "value": value}


def test_replay_validation_rejects_invalid_and_overlapping_bands():
    result = attention.normalize_replay([
        band(0, 10, 0.2), band(0, 10, 0.2), band(5, 15, 0.8), band(10, 20, 0.9),
        band(20, 10, 0.5), band(20, 30, float("nan")), band(30, 40, 2), {},
    ])
    assert result == [attention.ReplayInterval(0, 10, 0.2), attention.ReplayInterval(10, 20, 0.9)]


def test_interval_metrics_weight_overlap_and_count_chat_with_exclusive_end():
    context = attention.Attention(30, [attention.ReplayInterval(0, 10, 0.2),
                                       attention.ReplayInterval(10, 30, 0.8)],
                                  [3, 5, 10, 15], True, [Peak(10, 4, "audio"), Peak(20, 3, "audio")])
    result = context.interval(5, 15)
    assert result["replay_index_mean"] == 0.5
    assert result["replay_index_max"] == 0.8
    assert result["replay_coverage"] == 1
    assert result["chat_messages"] == 2
    assert result["chat_messages_per_minute"] == 12
    assert result["audio_peak_count"] == 1
    assert result["audio_peak_strength_max"] == 4


def test_partial_coverage_is_not_presented_as_full_coverage():
    context = attention.Attention(30, [attention.ReplayInterval(10, 20, 0.8)], [], False, [])
    result = context.interval(0, 20)
    assert result["replay_index_mean"] == 0.8
    assert result["replay_coverage"] == 0.5
    assert result["chat_messages"] is None


def test_no_access_data_does_not_invent_audience_metrics(tmp_path):
    context = attention.Attention(60, [], [], False, [])
    report = context.report(tmp_path)
    assert report["ranking_metric"] is None
    assert report["top_segments"] == []
    assert report["replay_available"] is False
    assert context.interval(0, 30)["replay_index_mean"] is None


def test_replay_peaks_are_relative_and_flat_curves_have_no_spikes():
    values = [0.1, 0.3, 0.9, 0.2, 1.0, 0.1]
    context = attention.Attention(180, [attention.ReplayInterval(i * 30, (i + 1) * 30, v)
                                        for i, v in enumerate(values)], [], False, [])
    peaks = context.replay_peaks()
    assert [(p.time, p.strength, p.kind) for p in peaks] == [(75, 0.9, "replay"), (135, 1, "replay")]
    flat = attention.Attention(60, [attention.ReplayInterval(0, 30, 0.5),
                                    attention.ReplayInterval(30, 60, 0.5)], [], False, [])
    assert flat.replay_peaks() == []


def test_missing_new_heatmap_preserves_valid_cache_for_same_video(tmp_path):
    path = attention.save_source({"id": "a", "heatmap": [band(0, 30, 0.5)]}, tmp_path, "url")
    before = path.read_bytes()
    attention.save_source({"id": "a", "heatmap": None}, tmp_path, "url")
    assert path.read_bytes() == before
    attention.save_source({"id": "b", "heatmap": None}, tmp_path, "different-url")
    assert json.loads(path.read_text())["available"] is False


def test_chat_peaks_are_recomputed_from_current_message_times():
    times = [float(t) for t in range(0, 1000, 10)] + [500.0] * 30
    context = attention.Attention(1000, [], sorted(times), True, [])
    assert any(p.kind == "chat" and p.time == 500 for p in context.chat_peaks())


def test_metrics_command_is_offline_without_llm_and_preserves_clips(tmp_path, monkeypatch):
    Transcript("pt", 60).save(tmp_path / "transcript.json")
    (tmp_path / "signals.json").write_text("[]")
    (tmp_path / "chat.json").write_text("[1, 5, 10, 31]")
    attention.save_source({"id": "a", "heatmap": [band(0, 30, 1), band(30, 60, 0.8)]}, tmp_path, "url")
    clips = tmp_path / "clips" / "existing"
    clips.mkdir(parents=True)
    video = clips / "clip.mp4"
    video.write_bytes(b"keep video")
    manifest = clips / "clips.json"
    manifest.write_text('[{"start": 0, "end": 10, "file": "clip.mp4"}]')
    before = manifest.read_bytes()

    def fail(*args, **kwargs):
        pytest.fail("metrics must not call an LLM or refresh the network without --source")

    monkeypatch.setattr(llm, "build", fail)
    monkeypatch.setattr(attention, "refresh_source", fail)
    assert main(["metrics", str(tmp_path)]) == 0
    report = json.loads((tmp_path / "attention.json").read_text())
    assert len(report["clips"]) == 1
    assert report["clips"][0]["metrics"]["chat_messages"] == 2
    assert report["top_uncovered_segments"][0]["start"] == 30
    assert manifest.read_bytes() == before
    assert video.read_bytes() == b"keep video"
    assert (tmp_path / "attention.md").is_file()
