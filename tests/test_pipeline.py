import json
import pytest
from dataclasses import replace

from clipper import attention, pipeline
from clipper.config import Settings
from clipper.detect import Highlight
from clipper.ingest import Source
from clipper.reframe import Framing
from clipper.transcribe import Transcript, Word


def test_rendering_new_batch_preserves_existing_clips(tmp_path, monkeypatch):
    original = tmp_path / "clips" / "longos"
    original.mkdir(parents=True)
    original_clip = original / "old.mp4"
    original_clip.write_bytes(b"existing clip")
    old_manifest = original / "clips.json"
    old_manifest.write_text('[{"start": 0, "end": 70}]')
    meta = tmp_path / "source.json"
    meta.write_text('{"original": true}')
    monkeypatch.setattr(pipeline.reframe, "plan", lambda *args: Framing("fit", 1920, 1080))

    def fake_render(video, start, end, framing, subs, output):
        output.write_bytes(b"new clip")
        return output

    monkeypatch.setattr(pipeline, "render_clip", fake_render)
    source = Source(tmp_path / "video.mp4", "video", tmp_path, None)
    pipeline.render_all(source, Transcript("pt", 200),
                        [Highlight(80, 150, 90, "new", "hook", "reason")],
                        Settings(), pipeline.Options(captions=False), "longos")
    assert original_clip.read_bytes() == b"existing clip"
    assert old_manifest.read_text() == '[{"start": 0, "end": 70}]'
    assert meta.read_text() == '{"original": true}'
    new_manifest = json.loads((tmp_path / "clips/longos-02/clips.json").read_text())
    assert new_manifest[0]["start"] == 80
    entry = new_manifest[0]
    publication = tmp_path / "clips/longos-02" / entry["publication_file"]
    assert publication.stem == (publication.parent / entry["file"]).stem
    assert publication.read_text(encoding="utf-8") == (
        "Título sugerido\nnew\n\nDescrição\nhook\n\nTrecho de “video”.\n\n"
        "Hashtags\n#ViraCortes #Shorts #Podcast\n"
    )
    assert pipeline.existing_ranges(tmp_path) == ((0, 70), (80, 150))


def test_publication_uses_approved_campaign_copy():
    highlight = Highlight(0, 24, 90, "Título original", "Gancho", "Motivo", campaign_review={
        "approved": True, "suggested_title": "Título da campanha",
        "caption": "Texto para publicar.", "hashtags": ["#Campanha", "campanha", "ViraCortes"],
    })
    text = pipeline.publication_text(highlight, "Vídeo", "gameplay", "pt")
    assert "Título da campanha\n\nDescrição\nTexto para publicar." in text
    assert text.endswith("#Campanha #ViraCortes\n")
    highlight.campaign_review["approved"] = False
    assert "#Campanha" not in pipeline.publication_text(highlight, "Vídeo", "gameplay", "pt")


def test_selection_cache_separates_duration_count_and_exclusions():
    options = pipeline.Options()
    original = pipeline.selection_id("model", options, ())
    assert pipeline.selection_id("model", replace(options, min_dur=60, max_dur=180), ()) != original
    assert pipeline.selection_id("model", replace(options, max_clips=6), ()) != original
    assert pipeline.selection_id("model", options, ((0, 70),)) != original


def test_pipeline_supplies_replay_hints_and_records_metrics_without_reasking_cached_llm(tmp_path, monkeypatch):
    engagement = attention.save_source({"id": "test", "heatmap": [
        {"start_time": 0, "end_time": 50, "value": 0.1},
        {"start_time": 50, "end_time": 100, "value": 0.9},
    ]}, tmp_path, "url")
    source = Source(tmp_path / "video.mp4", "video", tmp_path, None, engagement)
    transcript = Transcript("pt", 100, [Word("fala", i / 2, i / 2 + 0.4) for i in range(200)])
    monkeypatch.setattr(pipeline, "fetch", lambda *args: source)
    monkeypatch.setattr(pipeline, "transcribe", lambda *args: transcript)
    monkeypatch.setattr(pipeline.signals, "collect", lambda *args: [])
    calls = []

    def fake_llm(system, user, schema):
        assert "public replay popularity" in user
        calls.append(True)
        return {"highlights": [{"start": 40, "end": 80, "score": 85, "title": "cut"}]}

    monkeypatch.setattr(pipeline.llm, "build", lambda settings: fake_llm)
    monkeypatch.setattr(pipeline.reframe, "plan", lambda *args: Framing("fit", 1920, 1080))

    def fake_render(video, start, end, framing, subs, output):
        output.write_bytes(b"rendered")
        return output

    monkeypatch.setattr(pipeline, "render_clip", fake_render)
    options = pipeline.Options(max_clips=1, captions=False, batch="metrics-test")
    pipeline.run("url", tmp_path, Settings(), options)
    manifest = json.loads((tmp_path / "clips/metrics-test/clips.json").read_text())
    assert manifest[0]["engagement"]["replay_index_mean"] > 0.5
    assert manifest[0]["engagement"]["chat_messages"] is None
    assert list(tmp_path.glob("highlights.*attention-*.json"))
    pipeline.run("url", tmp_path, Settings(), options)
    assert len(calls) == 1
    assert (tmp_path / "clips/metrics-test-02/clips.json").is_file()


def test_speaker_focus_reaches_the_manifest_and_uses_a_separate_cache(tmp_path, monkeypatch):
    timeline = tmp_path / "speakers.json"
    timeline.write_text(json.dumps([
        {"start": 0, "end": 30, "speaker": "A"},
        {"start": 30, "end": 100, "speaker": "B"},
    ]))
    source = Source(tmp_path / "video.mp4", "video", tmp_path, None)
    transcript = Transcript("pt", 100, [Word("fala", i / 2, i / 2 + 0.4) for i in range(200)])
    monkeypatch.setattr(pipeline, "fetch", lambda *args: source)
    monkeypatch.setattr(pipeline, "transcribe", lambda *args: transcript)
    monkeypatch.setattr(pipeline.signals, "collect", lambda *args: [])

    def fake_llm(system, user, schema):
        assert "Speaker preference" in system
        assert '[speaker="A"]' in user
        return {"highlights": [
            {"start": 5, "end": 20, "score": 70, "title": "A"},
            {"start": 55, "end": 70, "score": 95, "title": "B"},
        ]}

    monkeypatch.setattr(pipeline.llm, "build", lambda settings: fake_llm)
    monkeypatch.setattr(pipeline.reframe, "plan", lambda *args: Framing("fit", 1920, 1080))

    def fake_render(video, start, end, framing, subs, output):
        output.write_bytes(b"rendered")
        return output

    monkeypatch.setattr(pipeline, "render_clip", fake_render)
    options = pipeline.Options(max_clips=2, min_dur=10, max_dur=30, captions=False, batch="focus",
                               prefer_speaker="A", speaker_turns=timeline, speaker_weight=0.5)
    pipeline.run("url", tmp_path, Settings(), options)
    manifest = json.loads((tmp_path / "clips/focus/clips.json").read_text())
    assert manifest[0]["title"] == "A"
    assert manifest[0]["score"] == 70
    assert manifest[0]["selection_score"] == 85
    assert manifest[0]["speaker_focus"]["speaker"] == "A"
    assert list(tmp_path.glob("highlights.*-speaker-*.json"))


def test_bad_speaker_name_is_rejected_before_download_or_transcription(tmp_path, monkeypatch):
    timeline = tmp_path / "speakers.json"
    timeline.write_text('[{"start": 0, "end": 30, "speaker": "A"}]')

    def fail(*args):
        pytest.fail("invalid speaker configuration must not start processing")

    monkeypatch.setattr(pipeline, "fetch", fail)
    options = pipeline.Options(prefer_speaker="missing", speaker_turns=timeline)
    with pytest.raises(ValueError, match="not present"):
        pipeline.run("url", tmp_path, Settings(), options)


@pytest.mark.parametrize("content_type", ["gameplay", "podcast"])
def test_template_metadata_and_credit_are_rendered_without_spoken_captions(tmp_path, monkeypatch, content_type):
    template = tmp_path / "template.json"
    template.write_text(json.dumps({"name": "brand", "background_mode": "framed",
                                    "title_enabled": True, "credit_enabled": True}), encoding="utf-8")
    monkeypatch.setattr(pipeline.reframe, "plan", lambda *args, **kwargs: Framing("gameplay", 1920, 1080))
    captured = {}

    def fake_render(video, start, end, framing, subs, output, *visual):
        captured["ass"] = subs.read_text(encoding="utf-8")
        captured["visual"] = visual
        output.write_bytes(b"video")
        return output

    monkeypatch.setattr(pipeline, "render_clip", fake_render)
    source = Source(tmp_path / "video.mp4", "video", tmp_path, None)
    transcript = Transcript("pt", 30, [Word("fala", 0, 1)])
    options = pipeline.Options(captions=False, template_file=template, credit="Fonte: Original")
    pipeline.render_all(source, transcript, [Highlight(0, 20, 90, "Título", "Hook", "Reason")],
                        Settings(), options, "template", content_type=content_type)
    events = [line for line in captured["ass"].splitlines() if line.startswith("Dialogue")]
    assert any(",TemplateCredit," in line for line in events)
    start = "0:00:05.20" if content_type == "gameplay" else "0:00:00.00"
    assert any(f",{start},0:00:20.00,TemplateTitle," in line for line in events)
    opening = [line for line in events if ",Main," in line]
    if content_type == "gameplay":
        assert len(opening) == 1 and "HOOK" in opening[0]
        assert opening[0].startswith("Dialogue: 1,0:00:00.00,0:00:05.20")
    else:
        assert opening == []
    assert captured["visual"] == (1080, 1920, "framed", "0x00E5FF@0.95")
    manifest = json.loads((tmp_path / "clips/template/clips.json").read_text(encoding="utf-8"))
    assert manifest[0]["template"]["name"] == "brand"
    assert manifest[0]["template"]["credit"] == "Fonte: Original"


def test_rerender_can_limit_to_top_clips(tmp_path, monkeypatch):
    (tmp_path / "source.json").write_text(json.dumps({"video": str(tmp_path / "video.mp4"),
                                                       "title": "video"}), encoding="utf-8")
    Transcript("pt", 60).save(tmp_path / "transcript.json")
    (tmp_path / "highlights.json").write_text(json.dumps([
        {"start": 0, "end": 20, "score": 50, "title": "low", "hook": "", "reason": ""},
        {"start": 20, "end": 40, "score": 90, "title": "high", "hook": "", "reason": ""},
    ]), encoding="utf-8")
    captured = []
    monkeypatch.setattr(pipeline, "render_all", lambda src, tx, clips, *args: captured.extend(clips))
    pipeline.rerender(tmp_path, Settings(), pipeline.Options(render_limit=1))
    assert [clip.title for clip in captured] == ["high"]
