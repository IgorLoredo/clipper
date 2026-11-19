import json
import hashlib
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from clipper import detect as detect_mod
from clipper import attention, campaign, llm, reframe, signals
from clipper.branding import Branding
from clipper.captions import CaptionStyle, build_ass, load_template
from clipper.codex_exchange import CodexExchange
from clipper.config import Settings
from clipper.detect import Highlight
from clipper.gameplay_selection import GameplayCriteria
from clipper.ingest import Source, fetch, slugify
from clipper.render import OUT_H, SPLIT_WEBCAM_HEIGHT, render_clip, render_preview
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
    visual_review: bool = True
    brand: Branding | None = None
    preview_only: bool = False
    preview_at: float | None = None
    render_limit: int | None = None
    gameplay_split: bool = False
    webcam_crop: tuple[int, int, int, int] | None = None
    select_only: bool = False
    gameplay_criteria: GameplayCriteria | None = None
    visual_review_llm: str = "codex"
    template_file: Path | None = None
    credit: str = ""


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
    if content_type == "gameplay":
        digest = hashlib.sha256(repr(options.gameplay_criteria or GameplayCriteria()).encode()).hexdigest()[:10]
        run_id += f"-gameplay-{digest}"
    if context.fingerprint:
        run_id += f"-attention-{context.fingerprint}"
    if preference:
        run_id += f"-speaker-{preference.fingerprint}"
    cache = src.workdir / f"highlights.{run_id}.json"
    visual_review = options.visual_review and content_type == "gameplay"
    shortlist_size = min(12, options.max_clips + 4) if visual_review else options.max_clips
    if visual_review:
        cache = cache.with_suffix(".shortlist.json")
    highlights = detect_mod.detect(
        transcript, peaks, llm.build(settings), shortlist_size,
        options.min_dur, options.max_dur, cache, excluded=excluded, preference=preference,
        window_seconds=options.llm_window_seconds, content_type=content_type,
        campaign_context=brief.summary() if brief else "",
        **({"gameplay_criteria": options.gameplay_criteria} if content_type == "gameplay" else {}),
    )
    if visual_review:
        from clipper.gameplay import review

        visual_settings = settings.with_llm(options.visual_review_llm)
        if visual_settings.llm_provider != "codex" or visual_settings.llm_fallbacks:
            raise ValueError("visual review currently requires --visual-review-llm codex[:model]")
        highlights = review(src.video, transcript, highlights, peaks, excluded,
                            options.min_dur, options.max_dur, options.max_clips,
                            CodexExchange(settings.codex_directory, visual_settings.model_for("codex")),
                            speaker_weight=preference.weight if preference else None,
                            campaign_context=brief.summary() if brief else "")
        run_id += f"-visual-{slugify(visual_settings.model_for('codex'), 30)}"
        cache = src.workdir / f"highlights.{run_id}.json"
        cache.write_text(json.dumps([asdict(clip) for clip in highlights], ensure_ascii=False, indent=2),
                         encoding="utf-8")
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
        latest.write_text(json.dumps([asdict(clip) for clip in highlights], ensure_ascii=False, indent=2),
                          encoding="utf-8")
    print(f"[run] {len(highlights)} highlights selected by {llm_id}", flush=True)
    if options.select_only:
        metadata = src.workdir / "source.json"
        if not metadata.exists():
            metadata.write_text(json.dumps({"video": str(src.video), "title": src.title}, ensure_ascii=False),
                                encoding="utf-8")
        print(f"[select] {cache}", flush=True)
    else:
        render_all(src, transcript, highlights, settings, options, options.batch or run_id, context, content_type)
    attention.save_report(context, src.workdir)
    return src.workdir


def rerender(workdir: Path, settings: Settings, options: Options) -> Path:
    meta = json.loads((workdir / "source.json").read_text(encoding="utf-8"))
    src = Source(video=Path(meta["video"]), title=meta["title"], workdir=workdir, chat=None)
    transcript = Transcript.load(workdir / "transcript.json")
    data = json.loads((workdir / "highlights.json").read_text(encoding="utf-8"))
    content_type = reframe.resolve_content_type(src.title, options.content_type)
    highlights = [Highlight(**h) for h in data]
    if options.render_limit is not None:
        highlights = sorted(highlights, key=lambda item: item.ranking_score, reverse=True)[:options.render_limit]
    render_all(src, transcript, highlights, settings, options,
               options.batch or "manual", None, content_type)
    return workdir


def publication_text(highlight: Highlight, source_title: str, content_type: str, language: str) -> str:
    review = highlight.campaign_review or {}
    approved = review if review.get("approved") else {}
    title = approved.get("suggested_title") or highlight.title.strip() or source_title
    description = approved.get("caption")
    if not description:
        hook = highlight.hook.strip()
        credit = f"Trecho de “{source_title}”." if language.lower().startswith("pt") else f"Clip from “{source_title}”."
        description = f"{hook}\n\n{credit}" if hook and hook.casefold() != title.casefold() else credit

    categories = {"gameplay": ("Gameplay", "Games"), "anime": ("Anime",),
                  "podcast": ("Podcast",), "mass": ("Missa",)}
    suggestions = approved.get("hashtags") or ("ViraCortes", "Shorts", *categories.get(content_type, ()))
    hashtags = {}
    for suggestion in suggestions:
        tag = "".join(suggestion.lstrip("#").split())
        if tag:
            hashtags.setdefault(tag.casefold(), f"#{tag}")
    return f"Título sugerido\n{title}\n\nDescrição\n{description}\n\nHashtags\n{' '.join(hashtags.values())}\n"


def render_all(src: Source, transcript: Transcript, highlights: list[Highlight],
               settings: Settings, options: Options, variant: str,
               attention_context: attention.Attention | None = None,
               content_type: str = "podcast") -> None:
    if options.gameplay_split and content_type != "gameplay":
        raise ValueError("split layout requires gameplay; use --content-type gameplay")
    source_meta = src.workdir / "source.json"
    if not source_meta.exists():
        source_meta.write_text(
            json.dumps({"video": str(src.video), "title": src.title}, ensure_ascii=False), encoding="utf-8")
    clips_dir = src.workdir / "clips" / next_variant(src.workdir, variant)
    clips_dir.mkdir(parents=True, exist_ok=False)
    print(f"[render] output {clips_dir}", flush=True)
    brand = options.brand if not options.gameplay_split and not options.template_file and options.brand and options.brand.enabled else None
    if brand:
        from clipper.brand_graphics import write_overlay

    overlay = write_overlay(brand, clips_dir / "viracortes.png") if brand and highlights and brand.mode != "horizontal" else None
    ordered = sorted(highlights, key=lambda x: x.ranking_score, reverse=True)
    ordered = ordered[:1 if options.preview_only else options.render_limit]
    manifest = []
    template = load_template(options.template_file)
    for rank, h in enumerate(ordered, 1):
        stem = f"{rank:02d}_{h.score:03d}_{slugify(h.title, 40)}"
        print(f"[render] {stem} ({h.start:.1f}s-{h.end:.1f}s)", flush=True)
        # Keep the established podcast call shape intact; gameplay opts into the new full-screen plan.
        if options.gameplay_split:
            framing = reframe.plan_gameplay_split(src.video, h.start, settings.face_model, options.webcam_crop)
        elif content_type == "gameplay" or (brand and brand.mode == "horizontal"):
            framing = reframe.plan(src.video, h.start, h.end, settings.face_model, content_type="gameplay")
        elif content_type in {"anime", "mass"}:
            framing = reframe.plan(src.video, h.start, h.end, settings.face_model, content_type=content_type)
        else:
            framing = reframe.plan(src.video, h.start, h.end, settings.face_model)
        aspect = framing.src_w / framing.src_h if framing.src_w > 0 and framing.src_h > 0 else 16 / 9
        if brand and brand.mode == "horizontal":
            overlay = write_overlay(brand, clips_dir / f"{stem}.overlay.png", aspect, h.title)
        subs = None
        overlay_text = ((template.title_enabled and bool(h.title.strip()))
                        or (template.brand_enabled and bool(template.brand.strip()))
                        or (template.credit_enabled and bool(options.credit.strip() or template.credit.strip())))
        if options.captions or overlay_text:
            subs = clips_dir / f"{stem}.ass"
            if options.template_file:
                captions = build_ass(transcript.words, h.start, h.end, template=template,
                                     title=h.title, credit=options.credit, include_captions=options.captions)
            elif options.gameplay_split:
                style = CaptionStyle(size=64, outline_width=5, margin_v=OUT_H - SPLIT_WEBCAM_HEIGHT - 26)
                captions = build_ass(transcript.words, h.start, h.end, style)
            elif brand:
                w, height = brand.size
                bottom, side = brand.caption_margins(aspect)
                style = CaptionStyle()
                style = replace(style, size=max(1, round(style.size * w / 1080)),
                                outline_width=max(1, round(style.outline_width * w / 1080)),
                                shadow=round(style.shadow * w / 1080), margin_v=round(height * bottom))
                captions = build_ass(transcript.words, h.start, h.end, style, w, height, round(w * side))
            else:
                captions = build_ass(transcript.words, h.start, h.end)
            subs.write_text(captions, encoding="utf-8")
        branding = {"brand": brand, "overlay": overlay} if brand else {}
        if options.template_file:
            branding = dict(width=template.width, height=template.height,
                            background_mode=template.background_mode, frame_color=template.frame_color)
        if options.preview_only or options.preview_at is not None:
            at = options.preview_at if options.preview_at is not None else min(1, (h.end - h.start) / 2)
            preview = render_preview(src.video, h.start, h.end, framing, subs,
                                     clips_dir / f"{stem}.preview.png", at=at, **branding)
            print(f"[preview] {preview}", flush=True)
        if options.preview_only:
            break
        output_path = clips_dir / f"{stem}.mp4"
        if options.template_file:
            output = render_clip(src.video, h.start, h.end, framing, subs, output_path,
                                 template.width, template.height, template.background_mode, template.frame_color)
        else:
            output = render_clip(src.video, h.start, h.end, framing, subs, output_path, **branding)
        publication = output.with_suffix(".txt")
        publication.write_text(publication_text(h, src.title, content_type, transcript.language), encoding="utf-8")
        manifest.append({**asdict(h), "file": output.name, "framing": framing.mode,
                         "content_type": content_type, "publication_file": publication.name})
        if options.template_file:
            manifest[-1]["template"] = {"name": template.name, "file": str(options.template_file),
                                         "credit": options.credit or template.credit}
        if attention_context is not None:
            manifest[-1]["engagement"] = attention_context.interval(h.start, h.end)
        if brand:
            manifest[-1]["branding"] = {"template": "viracortes", "mode": brand.mode, "size": brand.size}
        if framing.mode == "gameplay-split":
            manifest[-1]["webcam_crop"] = framing.webcam
        (clips_dir / "clips.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (clips_dir / "clips.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if content_type == "gameplay":
        print(f"\nGameplay | {len(manifest)} corte(s) em {clips_dir.resolve()}", flush=True)
        for item in manifest:
            start = int(item["start"])
            minutes, seconds = divmod(start, 60)
            duration = item["end"] - item["start"]
            print(f"  {minutes:02d}:{seconds:02d}  {duration:.0f}s  {item['title']}", flush=True)
