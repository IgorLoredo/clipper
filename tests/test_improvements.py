import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from clipper import (
    campaign,
    cli,
    codex_exchange,
    detect,
    ffmpeg,
    gameplay,
    gameplay_audio,
    gameplay_selection,
    ingest,
    llm,
    reframe,
    render,
    review_ui,
    signals,
    transcribe,
    usage,
)
from clipper.captions import CaptionStyle, CaptionTemplate, build_ass, dynamic_opening_duration
from clipper.config import Settings
from clipper.detect import Highlight
from clipper.signals import Peak
from clipper.transcribe import Transcript, Word


def test_usage_tracker_records_saves_and_reports(tmp_path, capsys):
    workdir = tmp_path / "demo-video"
    workdir.mkdir()
    tracker = usage.UsageTracker()
    tracker.record("gemini", "gemini-3.6-flash", input_tokens=1200, output_tokens=300, cached_tokens=400, total_tokens=1500)
    tracker.record("openai", "gpt-5-mini", input_tokens=800, output_tokens=200)
    tracker.save(workdir, tmp_path)

    saved = json.loads((workdir / "usage.json").read_text(encoding="utf-8"))
    assert saved["summary"]["totals"]["requests"] == 2
    assert saved["summary"]["totals"]["total_tokens"] == 2500
    assert saved["summary"]["totals"]["cached_tokens"] == 400
    assert (tmp_path / "_usage.jsonl").is_file()

    assert cli.main(["--out", str(tmp_path), "usage", "--days", "7"]) == 0
    out = capsys.readouterr().out
    assert "gemini:gemini-3.6-flash" in out
    assert "2,500" in out


def test_gemini_records_usage_and_supports_multimodal_images(tmp_path, monkeypatch):
    img = tmp_path / "frames" / "grid.jpg"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"\xff\xd8\xff\xd9")
    captured = {}

    class FakeClient:
        def __init__(self):
            def generate_content(**kwargs):
                captured.update(kwargs)
                meta = SimpleNamespace(
                    prompt_token_count=900,
                    candidates_token_count=150,
                    cached_content_token_count=200,
                    total_token_count=1050,
                )
                return SimpleNamespace(text='{"clips": []}', usage_metadata=meta)
            self.models = SimpleNamespace(generate_content=generate_content)

    monkeypatch.setattr(llm, "_require", lambda module, extra: SimpleNamespace(Client=FakeClient))
    settings = Settings(codex_directory=tmp_path).with_llm("gemini:gemini-vision")
    chain = llm.build(settings)
    result = chain("sys", "user", gameplay.SCHEMA, images=[{"index": 0, "file": "frames/grid.jpg"}])
    assert result == {"clips": []}
    assert isinstance(captured["contents"], list)
    assert any(isinstance(part, dict) and "inline_data" in part for part in captured["contents"])
    assert chain.usage.summary()["totals"]["total_tokens"] == 1050


def test_deepseek_uses_dynamic_schema_example(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            def create(**request):
                captured.update(request)
                return SimpleNamespace(
                    choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content='{"reviews": []}', refusal=None))],
                    usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120),
                )
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    monkeypatch.setattr(llm, "_require", lambda module, extra: SimpleNamespace(OpenAI=FakeOpenAI))
    chain = llm.build(Settings().with_llm("deepseek"))
    assert chain("sys", "user", gameplay_selection.SCHEMA) == {"reviews": []}
    assert 'Example JSON structure: {"reviews": []}' in captured["messages"][0]["content"]
    assert chain.usage.summary()["totals"]["total_tokens"] == 120


def test_llm_chain_retries_transient_errors_before_dropping():
    attempts = {"count": 0}

    def flaky(system, user, schema):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise TimeoutError("read timeout")
        return {"ok": True}

    chain = llm.LLMChain([("a:1", lambda: flaky)], max_retries=2, retry_delay=0.0)
    assert chain("s", "u", {"type": "object"}) == {"ok": True}
    assert attempts["count"] == 2
    assert chain.label == "a:1"


def test_codex_stage_distinguishes_gameplay_selection_from_campaign_review():
    assert codex_exchange._stage(gameplay_selection.SCHEMA) == "selection"
    assert codex_exchange._stage(campaign.REVIEW_SCHEMA) == "campaign_review"
    assert codex_exchange._stage(gameplay.SCHEMA) == "visual_review"
    assert codex_exchange._stage(detect.SCHEMA) == "selection"


def test_local_z_finds_peaks_in_compressed_audio_and_single_pass_levels(tmp_path):
    # Simulates compressed audio: many quiet pauses + a tight ceiling where global MAD hides spikes.
    rng = np.random.default_rng(7)
    levels = np.full(180, -14.0)
    levels[::4] = -35.0  # pauses inflate global MAD
    levels[60] = -4.0    # real local shout
    levels[130] = -3.5   # second local shout
    assert len(signals.find_peaks(levels, 1.0, "audio")) < 2
    local_peaks = signals.find_peaks(levels, 1.0, "audio", local_window=40)
    times = [p.time for p in local_peaks]
    assert any(abs(t - 60.0) <= 1 for t in times)
    assert any(abs(t - 130.0) <= 1 for t in times)

    # Single-pass levels extraction via gameplay_audio.analyze
    t = np.arange(gameplay_audio.RATE) / gameplay_audio.RATE
    chunks = [np.sin(2 * np.pi * 600 * t) * 1500 for _ in range(10)]
    chunks[5] = np.sin(2 * np.pi * 1000 * t) * 8000
    captured_levels: list[float] = []
    events = gameplay_audio.analyze(chunks, [Word("oi", i + 0.1, i + 0.8) for i in range(10)], 10,
                                    levels_out=captured_levels)
    assert len(captured_levels) == 10
    assert any(ev.features.get("voice_band_ratio", 0) > 0.8 for ev in events)


def test_reaction_clusters_surface_chat_replay_and_vocal_peaks_and_compact_format():
    units = gameplay_selection.speech_units([
        Word("Conversa calma sobre o mapa.", 10.0, 12.0),
        Word("Aqui acontece algo engraçado sem grito.", 40.0, 43.0),
        Word("Outro momento tranquilo.", 80.0, 82.0),
    ])
    peaks = [Peak(11.0, 3.5, "chat"), Peak(41.0, 0.92, "replay")]
    vocal = [gameplay_audio.AudioEvent(
        80.0, 81.0, 80.5, "pitch_change", 7.0, 0.9,
        {"pitch_z": 5.0, "pitch_confidence": 0.9, "onset_z": 4.0, "voice_band_ratio": 0.75},
    )]
    clusters = gameplay_selection.reaction_clusters(units, peaks, 10, 30, events=vocal)
    kinds = {sig[0] for c in clusters for sig in c["signals"]}
    assert {"chat_spike", "replay_peak", "vocal_peak"} <= kinds

    full = gameplay_selection.format_batch(clusters, units, None, 10, 30, compact=False)
    compact = gameplay_selection.format_batch(clusters, units, None, 10, 30, compact=True)
    assert len(compact) < len(full)
    parsed = json.loads(compact)
    assert all(len(row) == 4 for row in parsed["speech"])


def test_visual_review_refine_metadata_and_compact_prompts(tmp_path, monkeypatch):
    capture = Mock()
    capture.get.side_effect = lambda prop: 1 if prop == cv2.CAP_PROP_FPS else 60
    monkeypatch.setattr(gameplay.cv2, "VideoCapture", lambda path: capture)
    monkeypatch.setattr(gameplay, "_cached_visual_peaks", lambda *args: [])
    monkeypatch.setattr(gameplay, "_sheet", lambda *args: "digest")
    source = tmp_path / "source.mp4"
    source.touch()
    candidate = Highlight(0, 20, 75, "Título inicial", "Hook inicial", "Motivo inicial", selection_score=75.0)
    transcript = Transcript("pt", 60, [Word("Olha esse carro voando!", 1, 3)])

    def reviewer(system, user, schema, *, images):
        payload = json.loads(user)
        assert "transcript" not in payload[0]
        assert "timed_transcript" in payload[0]
        return {"clips": [{"index": 0, "score": 95, "title": "Carro decolou no meio da fuga",
                           "hook": "O CARRO SIMPLESMENTE DECOLOU", "reason": "Os quadros confirmam a capotagem"}]}

    result = gameplay.review(
        source, transcript, [candidate], [], (), 20, 40, 1, reviewer,
        speaker_weight=None, frames_dir=tmp_path, refine_metadata=True, compact_prompts=True,
    )
    assert len(result) == 1
    assert result[0].title == "Carro decolou no meio da fuga"
    assert result[0].hook == "O CARRO SIMPLESMENTE DECOLOU"
    assert result[0].selection_score == pytest.approx(83.0)
    assert result[0].score == 83


def test_detect_concurrent_windows(tmp_path):
    words = []
    for block in range(3):
        base = block * 840.0
        words += [Word("fala", base + 5 + i * 0.5, base + 5 + i * 0.5 + 0.4) for i in range(80)]
    transcript = Transcript("pt", 2400.0, words)

    def fake_llm(system, user, schema):
        first = float(user.split("[", 1)[1].split("]", 1)[0])
        return {"highlights": [{"start": first, "end": first + 30, "score": 80, "title": "T", "hook": "H", "reason": "R"}]}

    clips = detect.detect(transcript, [], fake_llm, 6, 10, 60, tmp_path / "h.json", concurrency=3)
    assert len(clips) == 3


def test_load_json3_parses_word_offsets(tmp_path):
    sub = tmp_path / "legendas.pt-orig.json3"
    sub.write_text(json.dumps({
        "events": [
            {"tStartMs": 1000, "dDurationMs": 2000, "segs": [
                {"utf8": "Olha"}, {"utf8": " isso", "tOffsetMs": 400}, {"utf8": " agora!", "tOffsetMs": 900},
            ]},
            {"tStartMs": 4000, "dDurationMs": 1500, "segs": [{"utf8": "Que loucura"}]},
        ]
    }), encoding="utf-8")
    tx = transcribe.load_json3(sub, duration=60.0, language="pt")
    assert tx.language == "pt" and tx.duration == 60.0
    assert [w.text for w in tx.words] == ["Olha", "isso", "agora!", "Que", "loucura"]
    assert tx.words[0].start == 1.0
    assert tx.words[1].start == 1.4
    assert tx.words[2].start == 1.9
    assert transcribe.find_json3(tmp_path, "pt") == sub


def test_ingest_retries_chat_when_stale_part_file_exists(tmp_path, monkeypatch):
    calls = []

    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def extract_info(self, url, download=False):
            return {"title": "Live", "id": "abc", "heatmap": []}
        def process_ie_result(self, info, download=True):
            calls.append(self.opts)
            if self.opts.get("skip_download"):
                Path(self.opts["outtmpl"].replace("%(ext)s", "live_chat.json")).write_text("[]", encoding="utf-8")

    monkeypatch.setattr("yt_dlp.YoutubeDL", FakeYDL)
    monkeypatch.setattr(ingest, "ffmpeg_exe", lambda: "ffmpeg")
    workdir = tmp_path / "live-abc"
    workdir.mkdir()
    (workdir / "source.mp4").touch()
    stale = workdir / "source.live_chat.json.part"
    stale.touch()

    source = ingest._download("https://example.invalid/live", tmp_path)
    assert not stale.exists()
    assert source.chat is not None and source.chat.name == "source.live_chat.json"
    assert len(calls) == 1 and calls[0]["skip_download"] is True


def test_stationary_webcam_clustering_and_cache(tmp_path):
    # Peripheral webcam consistently around (1550, 500) vs 1-frame moving NPC at (50, 50)
    detections = [
        (50, 50, 100, 100),
        (1550, 500, 120, 120),
        (1554, 502, 120, 120),
        (1548, 498, 120, 120),
    ]
    crop = reframe._pick_stationary_webcam(detections, 1920, 1080)
    assert crop is not None
    x, y, w, h = crop
    assert x > 1300 and y > 350 and w % 2 == 0 and h % 2 == 0

    video = tmp_path / "v.mp4"
    video.write_bytes(b"data")
    cache = tmp_path / "webcam.json"
    reframe._save_cached_webcam(cache, video, crop)
    assert reframe._load_cached_webcam(cache, video, 1920, 1080) == crop


def test_render_remembers_failed_gpu_encoder_and_supports_audio_norm_and_podcast_split(tmp_path, monkeypatch):
    attempted = []

    def fake_run(args, cwd=None):
        encoder = args[args.index("-c:v") + 1]
        attempted.append((encoder, "-af" in args))
        if cwd is not None and encoder != render.CPU_ENCODER[0]:
            raise ffmpeg.FFmpegError("GPU unavailable")

    monkeypatch.setattr(ffmpeg, "run", fake_run)
    render.preferred_encoder.cache_clear()
    try:
        render.render_clip(tmp_path / "in.mp4", 0, 10, reframe.Framing("fit", 320, 180),
                           None, tmp_path / "c1.mp4", normalize_audio=True)
        render.render_clip(tmp_path / "in.mp4", 0, 10, reframe.Framing("fit", 320, 180),
                           None, tmp_path / "c2.mp4", normalize_audio=False)
    finally:
        render.preferred_encoder.cache_clear()

    nvenc = render.HARDWARE_ENCODERS[0][0]
    cpu = render.CPU_ENCODER[0]
    # Clip 1 probes nvenc, tries nvenc (fails), retries cpu; Clip 2 goes straight to cpu!
    assert attempted == [(nvenc, False), (nvenc, True), (cpu, True), (cpu, False)]

    split_graph = render.video_filter(
        reframe.Framing("podcast-split", 1920, 1080, boxes=((0, 0, 800, 720), (1120, 0, 800, 720))),
        "clip's [1].ass",
    )
    assert "vstack" in split_graph
    assert r"ass='clip\'s \[1\].ass'" in split_graph


def test_captions_borderstyle_and_dynamic_opening_and_keep_captions():
    tpl = CaptionTemplate(title_enabled=True, title_box=True, credit_enabled=True, credit_box=False)
    words = [Word("oi", 0.5, 1.0), Word("pessoal", 1.1, 1.8), Word("fim", 4.0, 4.5)]
    dur = dynamic_opening_duration("Teste curto")
    assert 2.4 <= dur <= 4.8
    ass = build_ass(words, 0.0, 10.0, template=tpl, title="T", credit="C",
                    opening="Teste curto", opening_duration=dur, keep_captions_during_opening=True)
    assert "Style: TemplateTitle," in ass and ",3,3,1,8," in ass
    assert "Style: TemplateCredit," in ass and ",1,2,1,2," in ass
    dialogues = [line for line in ass.splitlines() if line.startswith("Dialogue:")]
    assert any("0:00:00.50" in line and "OI" in line for line in dialogues)


def test_review_ui_http_endpoints(tmp_path):
    video = tmp_path / "source.mp4"
    video.write_bytes(b"0123456789ABCDEF")
    (tmp_path / "source.json").write_text(json.dumps({"video": str(video), "title": "Demo"}), encoding="utf-8")
    (tmp_path / "highlights.json").write_text(json.dumps([
        {"start": 1.0, "end": 25.0, "score": 88, "title": "Corte 1", "hook": "Hook 1", "reason": "R"},
    ]), encoding="utf-8")

    app = review_ui.ReviewApp(tmp_path)
    status, _, body = app.handle("GET", "/")
    assert status == 200 and b"Clipper Review" in body

    status, _, body = app.handle("GET", "/api/state")
    assert status == 200
    state = json.loads(body.decode("utf-8"))
    assert state["title"] == "Demo" and len(state["highlights"]) == 1

    status, headers, body = app.handle("GET", "/video", headers={"Range": "bytes=4-9"})
    assert status == 206 and body == b"456789" and headers["Content-Range"] == "bytes 4-9/16"

    status, _, body = app.handle(
        "POST", "/api/highlights",
        body=json.dumps([{"start": 2.0, "end": 30.0, "score": 92, "title": "Editado", "hook": "Novo"}]).encode("utf-8"),
    )
    assert status == 200 and json.loads(body.decode("utf-8")) == {"saved": 1}
    saved = json.loads((tmp_path / "highlights.json").read_text(encoding="utf-8"))
    assert saved[0]["title"] == "Editado" and saved[0]["start"] == 2.0

    status, _, body = app.handle(
        "POST", "/api/webcam",
        body=json.dumps({"webcam_crop": [100, 200, 300, 200]}).encode("utf-8"),
    )
    assert status == 200 and json.loads(body.decode("utf-8")) == {"webcam_crop": [100, 200, 300, 200]}
    assert json.loads((tmp_path / "webcam.json").read_text(encoding="utf-8"))["webcam_crop"] == [100, 200, 300, 200]


def test_cli_allows_api_visual_review_only_with_opt_in(tmp_path, monkeypatch):
    captured = {}

    def fake_run(source, out, settings, options, chat):
        captured.update(settings=settings, options=options)
        return tmp_path

    monkeypatch.setattr("clipper.cli.run", fake_run)
    assert cli.main([
        "gameplay", "video.mp4", "--llm", "gemini:g1", "--visual-review-llm", "gemini:g2",
        "--allow-api-visual-review", "--visual-refine-all", "--compact-prompts",
        "--llm-concurrency", "3", "--normalize-audio", "--dynamic-opening",
        "--keep-opening-captions", "--preview-safe-zone", "--transcript-source", "auto",
        "--refine-captions",
    ]) == 0
    opts = captured["options"]
    assert opts.allow_api_visual_review is True
    assert opts.visual_refine_all is True
    assert opts.compact_prompts is True
    assert opts.llm_concurrency == 3
    assert opts.normalize_audio is True
    assert opts.dynamic_opening is True
    assert opts.keep_opening_captions is True
    assert opts.preview_safe_zone is True
    assert opts.transcript_source == "auto"
    assert opts.refine_captions is True
