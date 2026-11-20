import json

import pytest

from clipper.captions import CaptionStyle, ass_time, build_ass, group_words, load_template
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


def test_gameplay_opening_uses_yellow_and_delays_spoken_captions():
    style = CaptionStyle(size=64, margin_v=1174)
    transcript = [Word("antes", 1.0, 1.5), Word("voltou", 5.0, 5.5)]
    ass = build_ass(transcript, 0.0, 8.0, style, opening="As galinhas vão me dedurar?")
    opening, caption = [line for line in ass.splitlines() if line.startswith("Dialogue")]
    assert opening == (
        r"Dialogue: 1,0:00:00.00,0:00:05.20,Main,,0,0,1122,,"
        r"{\fs80\c&H00E5FF&}AS GALINHAS VÃO ME DEDURAR?"
    )
    assert caption.startswith("Dialogue: 0,0:00:05.20,0:00:05.50")
    assert "VOLTOU" in caption


def test_template_keeps_title_and_credit_when_spoken_captions_are_disabled(tmp_path):
    path = tmp_path / "template.json"
    path.write_text(json.dumps({"title_enabled": True, "credit_enabled": True,
                                "background_mode": "framed"}), encoding="utf-8")
    template = load_template(path)
    ass = build_ass(words(), 10, 14, template=template, title="Meu {título}",
                    credit="Fonte: canal", include_captions=False)
    dialogue = [line for line in ass.splitlines() if line.startswith("Dialogue")]
    assert len(dialogue) == 2
    assert "TemplateTitle" in dialogue[0] and "Meu (título)" in dialogue[0]
    assert "TemplateCredit" in dialogue[1] and "Fonte: canal" in dialogue[1]
    assert not any(",Main," in line for line in dialogue)


def test_template_rejects_unknown_fields_and_unsafe_frame_color(tmp_path):
    path = tmp_path / "template.json"
    path.write_text('{"typo": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown template fields"):
        load_template(path)
    path.write_text('{"frame_color": "red;movie=bad"}', encoding="utf-8")
    with pytest.raises(ValueError, match="frame_color"):
        load_template(path)
    path.write_text('{"caption": {"max_words": 0}}', encoding="utf-8")
    with pytest.raises(ValueError, match="max_words"):
        load_template(path)
    path.write_text('{"caption": {"font": "Arial,Injected"}}', encoding="utf-8")
    with pytest.raises(ValueError, match="font"):
        load_template(path)


def test_legacy_layout_field_remains_compatible(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text('{"layout": "framed"}', encoding="utf-8")
    template = load_template(path)
    assert template.layout == "framed" and template.background_mode == "framed"
