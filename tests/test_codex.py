import json
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from clipper import gameplay, pipeline
from clipper.detect import Highlight
from clipper.gameplay import _peak_clips, _visual_peaks
from clipper.cli import main
from clipper.signals import Peak
from clipper.transcribe import Transcript, Word


def answer(request, payload):
    response = request.with_suffix("").with_suffix(".response.json")
    response.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize(("spoken", "hook", "guidance"), [
    ("Esse salto dá para acertar de primeira!", "Era só acertar o primeiro salto",
     "rather than announcing the failure"),
    ("A ponte está tremendo. Será que aguenta?", "Essa ponte vai dar trabalho",
     "plausible consequence"),
    ("A trilha termina no topo da montanha.", "A trilha leva até o topo da montanha",
     "specific descriptive"),
    ("Só falta guardar as ferramentas.", "Era só guardar as ferramentas",
     "personal relationships"),
])
def test_visual_review_refines_ranked_hook_only_with_frames(tmp_path, monkeypatch, spoken, hook, guidance):
    capture = Mock()
    capture.get.side_effect = lambda prop: 1 if prop == cv2.CAP_PROP_FPS else 60
    monkeypatch.setattr(gameplay.cv2, "VideoCapture", lambda path: capture)
    monkeypatch.setattr(gameplay, "_cached_visual_peaks", lambda *args: [])
    sheets = iter(["digest", None])
    monkeypatch.setattr(gameplay, "_sheet", lambda *args: next(sheets))
    source = tmp_path / "source.mp4"
    source.touch()
    candidates = [
        Highlight(0, 20, 90, "Tentativa no salto", "Gancho anterior", "Motivo", selection_score=90.25),
        Highlight(25, 45, 80, "Outro salto", "Gancho preservado", "Motivo", selection_score=80.5),
    ]

    def answer_visual(system, user, schema, *, images):
        assert "separate from the publication title" in system
        assert "unchanged when the frames do not substantiate" in system
        assert guidance in system
        assert [image["index"] for image in images] == [0]
        assert json.loads(user)[0]["hook"] == "Gancho anterior"
        assert json.loads(user)[0]["transcript"] == spoken
        return {"clips": [
            {"index": 1, "title": "Ignore", "hook": "Sem apoio visual", "reason": "Ignore", "score": 99},
            {"index": 0, "title": "Ignore", "hook": hook, "reason": "Ignore", "score": 70},
        ]}

    exchange = Mock(side_effect=answer_visual, directory=tmp_path)
    transcript = Transcript("pt", 60, [Word(spoken, 0, 3)])
    result = gameplay.review(source, transcript, candidates, [], (), 20, 40, 2,
                             exchange, speaker_weight=None)
    assert [clip.hook for clip in result] == [hook, "Gancho preservado"]
    assert [clip.ranking_score for clip in result] == [90.25, 80.5]
    assert [clip.title for clip in result] == [clip.title for clip in candidates]


def test_visual_review_can_recover_a_candidate_with_frames(tmp_path, monkeypatch):
    capture = Mock()
    capture.get.side_effect = lambda prop: 1 if prop == cv2.CAP_PROP_FPS else 60
    monkeypatch.setattr(gameplay.cv2, "VideoCapture", lambda path: capture)
    monkeypatch.setattr(gameplay, "_cached_visual_peaks", lambda *args: [])
    monkeypatch.setattr(gameplay, "_sheet", lambda *args: "digest")
    source = tmp_path / "source.mp4"
    source.touch()
    transcript = Transcript("pt", 60, [Word("A ponte aguenta.", 10, 12), Word("Caiu.", 19, 20)])
    query = "Uma ponte quebra"

    def answer_visual(system, user, schema, *, images):
        payload = json.loads(user)
        assert payload["scene_query"] == query
        candidate, = payload["candidates"]
        assert candidate["opening_end"] == 15
        assert "aguenta" in candidate["opening_transcript"]
        assert "Caiu" not in candidate["opening_transcript"]
        assert "Caiu" in candidate["timed_transcript"]
        assert [image["index"] for image in images] == [0]
        return {"clips": [dict(index=0, score=81, title="A ponte", hook="A ponte aguenta?",
                               reason="Os quadros mostram a queda.")]}

    exchange = Mock(side_effect=answer_visual, directory=tmp_path)
    editorial = [{"visual_review_bounds": [10, 30], "reaction_times": [19], "final_score": 60}]
    clips = gameplay.review(source, transcript, [], [], (), 20, 40, 1, exchange,
                            speaker_weight=None, query=query, editorial=editorial)
    assert [(clip.start, clip.end, clip.ranking_score) for clip in clips] == [(10, 30, 81)]


def test_codex_exports_windows_and_resumes_through_campaign_review(tmp_path, monkeypatch, capsys):
    video = tmp_path / "video.mp4"
    video.touch()
    brief = tmp_path / "brief.json"
    brief.write_text(json.dumps({"name": "Example", "objective": "Explain an idea"}), encoding="utf-8")
    transcript = Transcript("pt", 179, [Word("fala.", i / 2, i / 2 + 0.4) for i in range(358)])
    monkeypatch.setattr(pipeline, "transcribe", lambda *args: transcript)
    monkeypatch.setattr(pipeline.signals, "collect", lambda *args: [])
    rendered = []
    monkeypatch.setattr(pipeline, "render_all", lambda src, tx, highlights, *args: rendered.append(highlights))
    args = ["--out", str(tmp_path / "out"), "run", str(video), "--llm", "codex",
            "--campaign", str(brief), "--llm-window-seconds", "120"]

    assert main(args) == 3
    directory = tmp_path / "out" / "video" / "codex-llm"
    status = json.loads((directory / "codex-status.json").read_text(encoding="utf-8"))
    assert status["state"] == "awaiting_response" and status["stage"] == "selection"
    assert status["command"][:2] == ["clipper", "--out"]
    requests = sorted(directory.glob("*.request.json"))
    assert len(requests) == 2
    first = next(path for path in requests
                 if "\n[0.0] " in json.loads(path.read_text(encoding="utf-8"))["user"])
    second = next(path for path in requests if path != first)
    answer(first, {"highlights": [
        {"start": 0, "end": 30, "score": 90, "title": "Cut", "hook": "Hook", "reason": "Reason"},
    ]})
    answer(second, {"highlights": "invalid"})
    assert main(args) == 3
    assert "invalid response" in capsys.readouterr().out
    assert not rendered

    answer(second, {"highlights": []})
    assert main(args) == 3
    review = next(path for path in directory.glob("*.request.json") if path not in requests)
    assert "reviews" in json.loads(review.read_text(encoding="utf-8"))["schema"]["properties"]
    assert not rendered
    answer(review, {"reviews": [
        {"index": 0, "approved": False, "score": 30, "reasons": ["Off brief"],
         "suggested_title": "", "caption": "", "hashtags": []},
    ]})
    assert main(args) == 0
    assert rendered == [[]]


def test_gameplay_reviews_local_frames_and_can_select_a_moment_without_speech(tmp_path, monkeypatch):
    video = tmp_path / "video.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 2, (64, 36))
    assert writer.isOpened()
    for index in range(160):
        writer.write(np.full((36, 64, 3), index, dtype=np.uint8))
    writer.release()
    transcript = Transcript("pt", 80, [
        Word("Meu Deus!" if i == 10 else "fala.", i / 2, i / 2 + 0.4) for i in range(60)
    ])
    monkeypatch.setattr(pipeline, "transcribe", lambda *args: transcript)
    monkeypatch.setattr(pipeline.signals, "collect", lambda *args: [Peak(58, 10, "audio")])
    rendered = []
    monkeypatch.setattr(pipeline, "render_all", lambda src, tx, highlights, *args: rendered.extend(highlights))
    args = ["--out", str(tmp_path / "out"), "gameplay", str(video), "--max-clips", "1", "--llm", "codex"]

    assert main(args) == 3
    directory = tmp_path / "out" / "video" / "codex-llm"
    selection = next(directory.glob("*.request.json"))
    candidate, = json.loads(json.loads(selection.read_text(encoding="utf-8"))["user"])["clusters"]
    answer(selection, {"reviews": [
        {"id": candidate["id"], "accepted": True, "first": 0, "development": 5, "payoff": 10, "last": 50,
         "narrative": 85, "intensity": 85, "entertainment": 85, "standalone": 85, "pace": 85,
         "expectation_break": 70, "title": "Spoken clip", "hook": "Hook", "reason": "Reason"},
    ]})
    assert main(args) == 3
    visual_path = next(path for path in directory.glob("*.request.json") if path != selection)
    visual = json.loads(visual_path.read_text(encoding="utf-8"))
    assert len(visual["images"]) == 2
    for image in visual["images"]:
        grid = cv2.imread(str(directory / image["file"]))
        assert grid.shape == (360, 1280, 3)
    clips = json.loads(visual["user"])
    silent = next(clip for clip in clips if not clip["transcript"])
    answer(visual_path, {"clips": [
        {"index": silent["index"], "score": 90, "title": "Visual payoff", "hook": "Action", "reason": "Payoff"},
    ]})
    assert main(args) == 0
    assert [clip.title for clip in rendered] == ["Visual payoff"]
    assert rendered[0].start == silent["start"]
    status = json.loads((directory / "codex-status.json").read_text(encoding="utf-8"))
    assert status["state"] == "complete" and status["pending_requests"] == []


def test_quiet_scene_change_creates_a_visual_candidate(tmp_path):
    video = tmp_path / "scene-change.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 2, (64, 36))
    assert writer.isOpened()
    for index in range(120):
        value = 0 if index < 60 else 255
        writer.write(np.full((36, 64, 3), value, dtype=np.uint8))
    writer.release()
    capture = cv2.VideoCapture(str(video))
    try:
        peaks = _visual_peaks(capture, 60)
    finally:
        capture.release()
    assert peaks and all(peak.kind == "visual" for peak in peaks)
    candidates = _peak_clips(peaks, 60, 20, 40, [])
    assert candidates and candidates[0].start <= 30 <= candidates[0].end
