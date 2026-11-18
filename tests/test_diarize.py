import pytest

from clipper.diarize import map_anchors, non_overlapping_turns


def test_overlapping_voice_regions_are_left_out_of_the_timeline():
    turns = non_overlapping_turns([
        {"start": 0, "end": 4, "speaker": "one"},
        {"start": 3, "end": 6, "speaker": "two"},
        {"start": 6, "end": 8, "speaker": "two"},
    ])
    assert turns == [
        {"start": 0.0, "end": 3.0, "speaker": "one"},
        {"start": 4.0, "end": 8.0, "speaker": "two"},
    ]


def test_reference_interval_maps_a_name_only_when_it_is_clear():
    source = [{"start": 0, "end": 8, "speaker": "speaker_00"},
              {"start": 8, "end": 10, "speaker": "speaker_01"}]
    mapped, evidence = map_anchors(source, ["Gustavo Machado=0:8"])
    assert mapped[0]["speaker"] == "Gustavo Machado"
    assert mapped[1]["speaker"] == "speaker_01"
    assert evidence[0]["reference_coverage"] == 1


def test_ambiguous_reference_interval_is_rejected():
    source = [{"start": 0, "end": 5, "speaker": "speaker_00"},
              {"start": 5, "end": 10, "speaker": "speaker_01"}]
    with pytest.raises(ValueError, match="ambiguous"):
        map_anchors(source, ["Gustavo Machado=0:10"])
