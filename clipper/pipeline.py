import json
import hashlib
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from clipper import detect as detect_mod
from clipper import attention, campaign, llm, reframe, signals
from clipper.captions import build_ass
from clipper.config import Settings
from clipper.detect import Highlight
from clipper.ingest import Source, fetch, slugify
from clipper.render import render_clip
from clipper.transcribe import Transcript, transcribe
from clipper.speakers import load_preference


@dataclass
class Options:
    max_clips: int = 8
    min_dur: float = 20.0
    max_dur: float = 75.0
    language: str | None = None
    captions: bool = True
    batch: str | None = None
    exclude_existing: bool = False
    prefer_speaker: str | None = None
    speaker_turns: Path | None = None
    speaker_mode: str = "prefer"
    speaker_weight: float = 0.25
    content_type: str = "auto"
    llm_window_seconds: float = 900.0
    campaign_file: Path | None = None
    review_llm: str | None = None
    review_mode: str = "strict"


def existing_ranges(workdir: Path) -> tuple[tuple[float, float], ...]:
    ranges = set()
    for manifest in sorted((workdir / "clips").glob("*/clips.json")):
        for item in json.loads(manifest.read_text(encoding="utf-8")):
            ranges.add((float(item["start"]), float(item["end"])))
    return tuple(sorted(ranges))


def selection_id(llm_id: str, options: Options, excluded: tuple[tuple[float, float], ...]) -> str:
    name = f"{llm_id}-{options.min_dur:g}-{options.max_dur:g}s-n{options.max_clips}"
    if excluded:
        digest = hashlib.sha256(json.dumps(excluded).encode()).hexdigest()[:10]
        name += f"-exclude-{digest}"
    return name


def next_variant(workdir: Path, variant: str) -> str:
    base = slugify(variant, 140)
    candidate, index = base, 2
    while (workdir / "clips" / candidate).exists():
        candidate = f"{base}-{index:02d}"
        index += 1
    return candidate


def run(source: str, out_root: Path, settings: Settings, options: Options, chat: str | None = None) -> Path:
    brief = campaign.load(options.campaign_file) if options.campaign_file else None
    if options.prefer_speaker:
        if options.speaker_turns is None:
            raise ValueError("speaker preference requires a speaker timeline; Whisper does not identify speakers")
        load_preference(options.speaker_turns, options.prefer_speaker, mode=options.speaker_mode,
                        weight=options.speaker_weight)
    src = fetch(source, out_root, chat)
    settings = replace(settings, codex_directory=src.workdir / "codex-llm")
    print(f"[run] workdir {src.workdir}", flush=True)
    transcript = transcribe(src.video, src.workdir / "transcript.json", settings, options.language)
    content_type = reframe.resolve_content_type(src.title, options.content_type)
    print(f"[layout] {content_type} ({'manual' if options.content_type != 'auto' else 'title detection'})", flush=True)
    preference = None
    if options.prefer_speaker:
        if options.speaker_turns is None:
            raise ValueError("speaker preference requires a speaker timeline; Whisper does not identify speakers")
        preference = load_preference(options.speaker_turns, options.prefer_speaker, transcript.duration,
                                     options.speaker_mode, options.speaker_weight)
    peaks = signals.collect(src.video, transcript.duration, src.chat, src.workdir / "signals.json")
    context = attention.load(transcript.duration, peaks, src.chat, src.engagement or src.workdir / "engagement.json")
    peaks = [p for p in peaks if p.kind != "chat"] + context.chat_peaks() + context.replay_peaks()
    peaks.sort(key=lambda p: p.time)
    attention.save_report(context, src.workdir)
    llm_id = slugify(f"{settings.llm_provider}-{settings.model_for(settings.llm_provider)}", 80)
    excluded = existing_ranges(src.workdir) if options.exclude_existing else ()
    run_id = selection_id(llm_id, options, excluded)
    if context.fingerprint:
        run_id += f"-attention-{context.fingerprint}"
    if preference:
        run_id += f"-speaker-{preference.fingerprint}"
    cache = src.workdir / f"highlights.{run_id}.json"
    highlights = detect_mod.detect(
        transcript, peaks, llm.build(settings), options.max_clips,
        options.min_dur, options.max_dur, cache, excluded=excluded, preference=preference,
        window_seconds=options.llm_window_seconds, content_type=content_type,
        campaign_context=brief.summary() if brief else "",
    )
    if brief:
        reviewer_settings = settings.with_llm(options.review_llm) if options.review_llm else settings
        review_key = f"campaign-review-{brief.fingerprint}-{hashlib.sha256(cache.read_bytes()).hexdigest()[:10]}.json"
        if any(provider == "codex" for provider, _ in reviewer_settings.llm_chain):
            reviewer_id = hashlib.sha256(repr(reviewer_settings.llm_chain).encode()).hexdigest()[:10]
            review_key = review_key.removesuffix(".json") + f"-{reviewer_id}.json"
        reviews = campaign.review(brief, highlights, transcript, llm.build(reviewer_settings), src.workdir / review_key)
        for highlight, assessment in zip(highlights, reviews):
            highlight.campaign_review = assessment
        approved = sum(bool(review["approved"]) for review in reviews)
        print(f"[campaign] {approved}/{len(highlights)} clips approved by {reviewer_settings.llm_provider}", flush=True)
        if options.review_mode == "strict":
            highlights = [highlight for highlight in highlights if highlight.campaign_review and highlight.campaign_review["approved"]]
            print(f"[campaign] strict mode will render {len(highlights)} approved clips", flush=True)
    latest = src.workdir / "highlights.json"
    if not options.batch or not latest.exists():
        shutil.copyfile(cache, latest)
    print(f"[run] {len(highlights)} highlights selected by {llm_id}", flush=True)
    render_all(src, transcript, highlights, settings, options, options.batch or run_id, context, content_type)
    attention.save_report(context, src.workdir)
    return src.workdir


def rerender(workdir: Path, settings: Settings, options: Options) -> Path:
    meta = json.loads((workdir / "source.json").read_text(encoding="utf-8"))
    src = Source(video=Path(meta["video"]), title=meta["title"], workdir=workdir, chat=None)
    transcript = Transcript.load(workdir / "transcript.json")
    data = json.loads((workdir / "highlights.json").read_text(encoding="utf-8"))
    content_type = reframe.resolve_content_type(src.title, options.content_type)
    render_all(src, transcript, [Highlight(**h) for h in data], settings, options,
               options.batch or "manual", None, content_type)
    return workdir


def render_all(src: Source, transcript: Transcript, highlights: list[Highlight],
               settings: Settings, options: Options, variant: str,
               attention_context: attention.Attention | None = None,
               content_type: str = "podcast") -> None:
    source_meta = src.workdir / "source.json"
    if not source_meta.exists():
        source_meta.write_text(
            json.dumps({"video": str(src.video), "title": src.title}, ensure_ascii=False), encoding="utf-8")
    clips_dir = src.workdir / "clips" / next_variant(src.workdir, variant)
    clips_dir.mkdir(parents=True, exist_ok=False)
    print(f"[render] output {clips_dir}", flush=True)
    manifest = []
    for rank, h in enumerate(sorted(highlights, key=lambda x: x.ranking_score, reverse=True), 1):
        stem = f"{rank:02d}_{h.score:03d}_{slugify(h.title, 40)}"
        print(f"[render] {stem} ({h.start:.1f}s-{h.end:.1f}s)", flush=True)
        # Keep the established podcast call shape intact; gameplay opts into the new full-screen plan.
        if content_type == "gameplay":
            framing = reframe.plan(src.video, h.start, h.end, settings.face_model, content_type="gameplay")
        else:
            framing = reframe.plan(src.video, h.start, h.end, settings.face_model)
        subs = None
        if options.captions:
            subs = clips_dir / f"{stem}.ass"
            subs.write_text(build_ass(transcript.words, h.start, h.end), encoding="utf-8")
        output = render_clip(src.video, h.start, h.end, framing, subs, clips_dir / f"{stem}.mp4")
        manifest.append({**asdict(h), "file": output.name, "framing": framing.mode,
                         "content_type": content_type})
        if attention_context is not None:
            manifest[-1]["engagement"] = attention_context.interval(h.start, h.end)
        (clips_dir / "clips.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (clips_dir / "clips.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
