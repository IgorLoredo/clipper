import json

import cv2
import numpy as np

from clipper import pipeline
from clipper.cli import main
from clipper.signals import Peak
from clipper.transcribe import Transcript, Word


def answer(request, payload):
    response = request.with_suffix("").with_suffix(".response.json")
    response.write_text(json.dumps(payload), encoding="utf-8")


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
    transcript = Transcript("pt", 80, [Word("fala.", i / 2, i / 2 + 0.4) for i in range(60)])
    monkeypatch.setattr(pipeline, "transcribe", lambda *args: transcript)
    monkeypatch.setattr(pipeline.signals, "collect", lambda *args: [Peak(58, 10, "audio")])
    rendered = []
    monkeypatch.setattr(pipeline, "render_all", lambda src, tx, highlights, *args: rendered.extend(highlights))
    args = ["--out", str(tmp_path / "out"), "gameplay", str(video), "--max-clips", "1"]

    assert main(args) == 3
    directory = tmp_path / "out" / "video" / "codex-llm"
    selection = next(directory.glob("*.request.json"))
    answer(selection, {"highlights": [
        {"start": 0, "end": 25, "score": 85, "title": "Spoken clip", "hook": "Hook", "reason": "Reason"},
    ]})
    assert main(args) == 3
    visual_path = next(path for path in directory.glob("*.request.json") if path != selection)
    visual = json.loads(visual_path.read_text(encoding="utf-8"))
    assert len(visual["images"]) == 2
    for image in visual["images"]:
        grid = cv2.imread(str(directory / image["file"]))
        assert grid.shape == (360, 640, 3)
    clips = json.loads(visual["user"])
    silent = next(clip for clip in clips if not clip["transcript"])
    answer(visual_path, {"clips": [
        {"index": silent["index"], "score": 90, "title": "Visual payoff", "hook": "Action", "reason": "Payoff"},
    ]})
    assert main(args) == 0
    assert [clip.title for clip in rendered] == ["Visual payoff"]
    assert rendered[0].start == silent["start"]
