from clipper.captions import ass_time, build_ass, group_words
from clipper.transcribe import Word


def words():
    return [Word("olha", 10.0, 10.3), Word("isso", 10.3, 10.6), Word("aqui,", 10.6, 11.0),
            Word("gente", 11.0, 11.4), Word("{x}", 13.0, 13.4)]


def test_ass_time_format():
    assert ass_time(0) == "0:00:00.00"
    assert ass_time(3725.456) == "1:02:05.46"


def test_group_words_breaks_on_punctuation_and_gap():
    groups = group_words(words(), max_words=3, max_gap=0.6)
    assert [[w.text for w in g] for g in groups] == [["olha", "isso", "aqui,"], ["gente"], ["{x}"]]


def test_build_ass_relative_times_highlight_and_escape():
    ass = build_ass(words(), clip_start=10.0, clip_end=14.0)
    dialogue = [line for line in ass.splitlines() if line.startswith("Dialogue")]
    assert len(dialogue) == 5
    assert dialogue[0].startswith("Dialogue: 0,0:00:00.00,0:00:00.30")
    assert "OLHA" in dialogue[0] and "\\c&H00E5FF&" in dialogue[0]
    assert "(X)" in dialogue[-1]
    assert "PlayResY: 1920" in ass


def test_build_ass_skips_words_outside_clip():
    ass = build_ass(words(), clip_start=12.5, clip_end=14.0)
    assert sum(line.startswith("Dialogue") for line in ass.splitlines()) == 1
