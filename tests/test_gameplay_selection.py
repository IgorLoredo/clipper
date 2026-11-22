import json

import pytest

from clipper import cli, detect
from clipper.gameplay_selection import GameplayCriteria, reaction_clusters, speech_units
from clipper.transcribe import Transcript, Word


def test_gameplay_groups_reactions_and_ranks_complete_scenes(tmp_path):
    words = [
        Word("Vou tentar saltar o muro.", 0, 2),
        Word("Cê viu?", 5, 6),
        Word("Caraca, consegui!", 8, 9),
        Word("Cheguei do outro lado.", 12, 14),
        Word("Eita!", 36, 37),
        Word("Vou abrir o baú.", 60, 62),
        Word("Como assim?", 65, 66),
        Word("[risadas]", 69, 70),
        Word("Era a chave que faltava.", 72, 74),
    ]
    units = speech_units(words)
    clusters = reaction_clusters(units, [], 10, 20)
    assert [(item["first"], item["last"]) for item in clusters] == [(1, 2), (4, 4), (6, 7)]
    assert ("surprise", "ce viu") in clusters[0]["signals"]

    def review(system, user, schema):
        assert "isolated shouting" in system
        assert schema["required"] == ["reviews"]
        assert len(json.loads(user)["clusters"]) == 3
        return {"reviews": [
            dict(id=i, accepted=accepted, first=first, development=peak, payoff=peak, last=last,
                 title="Título", hook="Gancho", reason=reason, narrative=score, intensity=score,
                 entertainment=score, standalone=score, pace=score, expectation_break=score)
            for i, accepted, first, peak, last, score, reason in [
                (0, True, 0, 2, 3, 80, "O salto tem consequência"),
                (1, False, -1, -1, -1, 0, "Exclamação isolada"),
                (2, True, 5, 7, 8, 95, "O baú resolve a busca"),
            ]
        ]}

    cache = tmp_path / "highlights.json"
    clips = detect.detect(Transcript("pt", 90, words), [], review, 1, 10, 20, cache, content_type="gameplay")
    assert [(clip.start, clip.end, clip.ranking_score) for clip in clips] == [(60, 74, 95)]
    report = json.loads(cache.with_suffix(".candidates.json").read_text(encoding="utf-8"))
    assert {item["id"]: item["status"] for item in report["candidates"]}[1] == "Exclamação isolada"


def test_gameplay_without_reactions_skips_editorial_call(tmp_path):
    cache = tmp_path / "highlights.json"
    transcript = Transcript("pt", 90, [Word("A porta está fechada.", 0, 1)])

    def unexpected(*args):
        raise AssertionError("There is no gameplay candidate to review")

    assert detect.detect(transcript, [], unexpected, 4, 10, 20, cache, content_type="gameplay") == []
    report = json.loads(cache.with_suffix(".candidates.json").read_text(encoding="utf-8"))
    assert report["status"] == "no_qualified_candidates"


def test_podcast_uses_original_highlights_schema(tmp_path):
    transcript = Transcript("pt", 60, [Word("fala.", i, i + 0.5) for i in range(45)])

    def review(system, user, schema):
        assert "elite short-form video editor" in system
        assert schema is detect.SCHEMA
        assert user.startswith("Transcript:")
        return {"highlights": [{"start": 0, "end": 15, "score": 80, "title": "Título",
                                "hook": "Gancho", "reason": "Motivo"}]}

    cache = tmp_path / "highlights.json"
    clips = detect.detect(transcript, [], review, 1, 10, 20, cache, content_type="podcast")
    assert len(clips) == 1
    assert not cache.with_suffix(".candidates.json").exists()


@pytest.mark.parametrize("score", [60, 80])
def test_gameplay_query_checks_calm_speech_and_preserves_bounds_for_review(tmp_path, score):
    query = "Ele celebra antes de perder."
    transcript = Transcript("pt", 60, [
        Word("A partida acabou, ganhamos.", 0, 2),
        Word("Perdemos na verdade.", 5, 7),
        Word("Eu comemorei cedo demais.", 10, 12),
    ])

    def review(system, user, schema):
        request = json.loads(user)
        assert request["scene_query"] == query
        cluster, = request["clusters"]
        assert cluster["query_window"] and not cluster["reaction_times"]
        return {"reviews": [dict(
            id=cluster["id"], accepted=True, first=0, development=0, payoff=1, last=2,
            title="Celebrou cedo", hook="A partida parecia decidida", reason="O placar vira",
            **dict.fromkeys(GameplayCriteria().weights, score),
        )]}

    cache = tmp_path / "highlights.json"
    clips = detect.detect(transcript, [], review, 1, 10, 20, cache,
                          content_type="gameplay", query=query)
    entry, = json.loads(cache.with_suffix(".candidates.json").read_text(encoding="utf-8"))["candidates"]
    if score == 60:
        assert not clips
        assert entry["visual_review_bounds"] == [0, 12]
    else:
        assert [(clip.start, clip.end) for clip in clips] == [(0, 12)]


def test_weights_can_be_overridden():
    scores = dict(narrative=100, intensity=0, entertainment=0, standalone=0, pace=0, expectation_break=0)
    assert GameplayCriteria().score(scores) == 15
    assert GameplayCriteria(narrative=100, intensity=0, entertainment=0, standalone=0, pace=0,
                            expectation_break=0).score(scores) == 100


def test_expectation_break_improves_rank_without_being_required():
    criteria = GameplayCriteria()
    scores = dict(narrative=80, intensity=80, entertainment=80, standalone=80, pace=80, expectation_break=0)
    assert criteria.score(scores) >= criteria.minimum_score
    assert criteria.score(scores | {"expectation_break": 90}) == pytest.approx(criteria.score(scores) + 15.3)


@pytest.mark.parametrize(("command", "expected"), [
    (["gameplay", "video.mp4"], "gameplay"),
    (["run", "video.mp4", "--content-type", "podcast"], "podcast"),
])
def test_cli_passes_content_type_and_select_only(monkeypatch, tmp_path, command, expected):
    def run(source, out, settings, options, chat):
        assert options.content_type == expected
        assert options.select_only
        assert options.max_dur == (105 if expected == "gameplay" else 75)
        return tmp_path

    monkeypatch.setattr(cli, "run", run)
    assert cli.main([*command, "--select-only"]) == 0
