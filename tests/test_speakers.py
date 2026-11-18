import json

import pytest

from clipper.cli import main
from clipper.detect import detect, format_lines
from clipper.speakers import SpeakerPreference, Turn, load_preference
from clipper.transcribe import Transcript, Word


def transcript():
    return Transcript("pt", 100, [Word("fala", i / 2, i / 2 + 0.4) for i in range(200)])


def test_speaker_label_does_not_use_a_name_mentioned_in_text():
    focus = SpeakerPreference("A", (Turn(0, 1, "B"), Turn(1, 2, "A")))
    words = [Word("A", 0, 0.4), Word("respondeu", 1, 1.4)]
    lines = format_lines(words, preference=focus).splitlines()
    assert lines == ['[0.0] [speaker="B"] A', '[1.0] [speaker="A"] respondeu']
    assert focus.label(Word("incerto", 0.8, 1.2)) == "unknown"
    assert focus.label(Word("fora", 3, 4)) == "unknown"


@pytest.mark.parametrize("items", [
    [{"start": 10, "end": 5, "speaker": "A"}],
    [{"start": 0, "end": 110, "speaker": "A"}],
    [{"start": 0, "end": float("nan"), "speaker": "A"}],
    [{"start": 0, "end": 10, "speaker": "A"}, {"start": 9, "end": 15, "speaker": "B"}],
    [{"start": 0, "end": 10, "speaker": "unknown"}],
    [{"start": 0, "end": 10, "speaker": "B"}],
])
def test_invalid_or_ambiguous_timeline_is_rejected(tmp_path, items):
    path = tmp_path / "speakers.json"
    path.write_text(json.dumps(items))
    with pytest.raises(ValueError):
        load_preference(path, "A", 100)


def test_preference_ranks_speech_share_separately_from_editorial_quality(tmp_path):
    focus = SpeakerPreference("A", (Turn(0, 20, "A"), Turn(20, 100, "B")), weight=0.5)

    def fake(system, user, schema):
        assert "independent of this preference" in system
        assert '[speaker="A"]' in user and '[speaker="B"]' in user
        return {"highlights": [
            {"start": 0, "end": 15, "score": 70, "title": "A"},
            {"start": 30, "end": 45, "score": 95, "title": "B"},
        ]}

    result = detect(transcript(), [], fake, 2, 10, 30, tmp_path / "h.json", preference=focus)
    assert [h.title for h in result] == ["A", "B"]
    assert result[0].score == 70
    assert result[0].selection_score == 85
    assert result[0].speaker_focus["preferred_speech_share"] == 1
    assert result[1].speaker_focus["eligible"] is True


def test_exclusive_mode_rejects_other_and_unknown_speech(tmp_path):
    focus = SpeakerPreference("A", (Turn(0, 20, "A"), Turn(20, 50, "B")), mode="only")

    def fake(*args):
        return {"highlights": [
            {"start": 0, "end": 15, "score": 70, "title": "A"},
            {"start": 10, "end": 30, "score": 99, "title": "mixed"},
            {"start": 30, "end": 45, "score": 95, "title": "B"},
            {"start": 70, "end": 85, "score": 90, "title": "unknown"},
        ]}

    result = detect(transcript(), [], fake, 4, 10, 30, tmp_path / "h.json", preference=focus)
    assert [h.title for h in result] == ["A"]
    assert result[0].speaker_focus["other_speech_seconds"] == 0


def test_focus_change_does_not_reuse_another_participants_cache(tmp_path):
    turns = (Turn(0, 20, "A"), Turn(20, 100, "B"))
    calls = []

    def fake(*args):
        calls.append(True)
        return {"highlights": []}

    path = tmp_path / "h.json"
    detect(transcript(), [], fake, 2, 10, 30, path, preference=SpeakerPreference("A", turns))
    detect(transcript(), [], fake, 2, 10, 30, path, preference=SpeakerPreference("B", turns))
    assert len(calls) == 2


def test_cli_requires_timed_labels_before_processing(capsys):
    with pytest.raises(SystemExit):
        main(["run", "video.mp4", "--prefer-speaker", "A"])
    assert "no speaker identification" in capsys.readouterr().err


def test_exclusive_mode_skips_windows_without_the_preferred_speaker(tmp_path):
    words = [Word("fala", base + i / 2, base + i / 2 + 0.4)
             for base in (0, 900, 1700) for i in range(80)]
    source = Transcript("pt", 2000, words)
    focus = SpeakerPreference("A", (Turn(900, 940, "A"),), mode="only")
    calls = []

    def fake(*args):
        calls.append(True)
        return {"highlights": []}

    detect(source, [], fake, 2, 10, 30, tmp_path / "h.json", preference=focus)
    assert len(calls) == 1


def test_cli_rejects_exclusive_mode_without_a_speaker(capsys):
    with pytest.raises(SystemExit):
        main(["run", "video.mp4", "--speaker-mode", "only"])
    assert "requires --prefer-speaker" in capsys.readouterr().err
