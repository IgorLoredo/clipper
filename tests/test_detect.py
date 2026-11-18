from clipper.detect import Highlight, dedupe, detect, progress_path, format_lines, format_hints, parse_highlights, snap, windows
import json

import pytest

from clipper.llm import LLMChain, ProvidersExhausted
from clipper.signals import Peak
from clipper.transcribe import Transcript, Word


def make_words(text: str, start: float = 0.0, step: float = 0.5) -> list[Word]:
    return [Word(text=t, start=start + i * step, end=start + i * step + 0.4) for i, t in enumerate(text.split())]


def hl(start, end, score):
    return Highlight(start=start, end=end, score=score, title="t", hook="h", reason="r")


def test_windows_short_video_is_single_span():
    assert windows(300) == [(0.0, 300)]


def test_windows_long_video_overlap_and_cover_end():
    spans = windows(2000, size=900, overlap=60)
    assert spans[0] == (0.0, 900)
    assert spans[1][0] == 840
    assert spans[-1][1] == 2000


def test_format_lines_breaks_on_sentence_end():
    text = format_lines(make_words("Olha isso. Ninguém fala disso!"))
    assert text.splitlines() == ["[0.0] Olha isso.", "[1.0] Ninguém fala disso!"]


def test_snap_aligns_to_words_and_extends_to_sentence_end():
    words = make_words(" ".join(["palavra"] * 20) + " fim.")
    start, end = snap(1.2, 8.0, words, min_dur=5, max_dur=60)
    assert start == 0.85
    assert end == round(words[-1].end + 0.25, 2)


def test_snap_rejects_too_short():
    assert snap(0, 2, make_words("a b c d e f"), min_dur=10, max_dur=60) is None


def test_snap_truncates_to_max_duration():
    words = make_words(" ".join(["w"] * 200))
    start, end = snap(0, 100, words, min_dur=5, max_dur=30, extend_limit=0)
    assert end - start <= 30


def test_dedupe_keeps_best_of_overlapping():
    kept = dedupe([hl(0, 30, 50), hl(5, 35, 90), hl(100, 130, 40)])
    assert [h.score for h in kept] == [90, 40]


def test_parse_highlights_drops_invalid_and_clamps_score():
    parsed = parse_highlights({"highlights": [
        {"start": 10, "end": 5, "score": 80},
        {"start": "x", "end": 9},
        {"start": 1, "end": 30, "score": 150, "title": " T ", "hook": "H", "reason": "R"},
    ]})
    assert len(parsed) == 1
    assert parsed[0].score == 100 and parsed[0].title == "T"


def test_detect_passes_signal_hints_and_snaps(tmp_path):
    words = make_words(" ".join(["fala"] * 79) + " acabou.", step=0.5)
    transcript = Transcript(language="pt", duration=40.0, words=words)
    seen = {}

    def fake_llm(system, user, schema):
        seen["system"], seen["user"] = system, user
        return {"highlights": [{"start": 0.1, "end": 35, "score": 77, "title": "T", "hook": "H", "reason": "R"}]}

    result = detect(transcript, [Peak(time=12, strength=3.1, kind="audio")], fake_llm,
                    max_clips=3, min_dur=10, max_dur=60, cache=tmp_path / "h.json")
    assert "audio energy spikes" in seen["user"] and "12s" in seen["user"]
    assert "(pt)" in seen["system"]
    assert len(result) == 1 and result[0].score == 77
    assert (tmp_path / "h.json").is_file()


def _long_transcript():
    words = []
    for block in range(3):
        base = block * 840.0
        words += make_words(" ".join(["fala"] * 79) + " acabou.", start=base + 5, step=0.5)
    return Transcript(language="pt", duration=2400.0, words=words)


def _answer(user):
    first = float(user.split("[", 1)[1].split("]", 1)[0])
    return {"highlights": [{"start": first, "end": first + 35, "score": 70, "title": "T", "hook": "H", "reason": "R"}]}


def test_detect_resumes_from_saved_chunks_when_all_llms_fail(tmp_path):
    transcript = _long_transcript()
    cache = tmp_path / "h.json"
    asked = []

    def first_two(system, user, schema):
        asked.append(user)
        if len(asked) > 2:
            raise RuntimeError("insufficient_quota")
        return _answer(user)

    with pytest.raises(ProvidersExhausted, match="chunk 3/3.*resume"):
        detect(transcript, [], LLMChain([("a:1", lambda: first_two)]), 8, 10, 60, cache)
    saved = json.loads(progress_path(cache).read_text(encoding="utf-8"))["windows"]
    assert len(saved) == 2 and not cache.exists()

    second = []

    def other(system, user, schema):
        second.append(user)
        return _answer(user)

    result = detect(transcript, [], LLMChain([("b:2", lambda: other)]), 8, 10, 60, cache)
    assert len(second) == 1
    assert sorted(h.llm for h in result) == ["a:1", "a:1", "b:2"]
    assert cache.is_file()


def test_detect_falls_back_mid_video_without_redoing_chunks(tmp_path):
    transcript = _long_transcript()
    asked = {"a": 0, "b": 0}

    def quota_after_one(system, user, schema):
        asked["a"] += 1
        if asked["a"] > 1:
            raise RuntimeError("429 quota")
        return _answer(user)

    def backup(system, user, schema):
        asked["b"] += 1
        return _answer(user)

    chain = LLMChain([("a:1", lambda: quota_after_one), ("b:2", lambda: backup)])
    result = detect(transcript, [], chain, 8, 10, 60, tmp_path / "h.json")
    assert asked == {"a": 2, "b": 2}
    assert sorted(h.llm for h in result) == ["a:1", "b:2", "b:2"]


def test_progress_from_other_settings_is_ignored(tmp_path):
    transcript = _long_transcript()
    cache = tmp_path / "h.json"
    detect(transcript, [], lambda s, u, sc: _answer(u), 8, 10, 60, cache)
    cache.unlink()
    asked = []
    detect(transcript, [], lambda s, u, sc: asked.append(u) or _answer(u), 8, 10, 50, cache)
    assert len(asked) == 3


def test_completed_cache_is_not_reused_after_duration_change(tmp_path):
    words = make_words(" ".join(["fala"] * 200))
    transcript = Transcript("pt", 100, words)
    calls = []

    def fake_llm(*args):
        calls.append(True)
        return {"highlights": [{"start": 0, "end": 30, "score": 90}]}

    cache = tmp_path / "h.json"
    assert len(detect(transcript, [], fake_llm, 3, 10, 60, cache)) == 1
    assert detect(transcript, [], fake_llm, 3, 40, 60, cache) == []
    assert len(calls) == 2


def test_detect_avoids_existing_clips_and_includes_prompt_hint(tmp_path):
    words = make_words(" ".join(["fala"] * 240))
    transcript = Transcript("pt", 120, words)

    def fake_llm(system, user, schema):
        assert "already became clips" in user
        return {"highlights": [
            {"start": 0, "end": 30, "score": 99},
            {"start": 60, "end": 90, "score": 80},
        ]}

    result = detect(transcript, [], fake_llm, 3, 10, 60, tmp_path / "h.json", excluded=((0, 30),))
    assert len(result) == 1
    assert result[0].score == 80


def test_replay_hints_describe_relative_popularity_without_claiming_views():
    hint = format_hints([Peak(50, 0.9, "replay")], 0, 100)
    assert "relative index" in hint and "NOT view counts" in hint
    assert "50s" in hint and "0.90" in hint


def test_completed_cache_is_invalidated_when_attention_signals_change(tmp_path):
    transcript = Transcript("pt", 100, make_words(" ".join(["fala"] * 200)))
    calls = []

    def fake_llm(*args):
        calls.append(True)
        return {"highlights": []}

    path = tmp_path / "h.json"
    detect(transcript, [], fake_llm, 3, 10, 60, path)
    detect(transcript, [Peak(50, 1, "replay")], fake_llm, 3, 10, 60, path)
    assert len(calls) == 2
