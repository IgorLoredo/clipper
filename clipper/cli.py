import argparse
import sys
import math
from pathlib import Path

from clipper.codex_exchange import CodexResponseRequired
from clipper.branding import load_branding
from clipper.config import Settings, load_dotenv
from clipper.llm import PROVIDERS, ProvidersExhausted
from clipper.pipeline import Options, rerender, run
from clipper.attention import analyze


def _webcam_crop(value: str) -> tuple[int, int, int, int]:
    try:
        x, y, w, h = map(int, value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected X,Y,W,H in source pixels") from exc
    if min(x, y) < 0 or min(w, h) < 2:
        raise argparse.ArgumentTypeError("X and Y must be nonnegative, W and H must be at least 2")
    return x, y, w, h


def _add_render_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-captions", action="store_true", help="do not burn captions")
    identity = parser.add_mutually_exclusive_group()
    identity.add_argument("--viracortes", dest="viracortes", action="store_true", default=True,
                          help="apply the ViraCortes identity (default)")
    identity.add_argument("--no-viracortes", dest="viracortes", action="store_false",
                          help="export without the ViraCortes identity")
    parser.add_argument("--template-config", type=Path, help="custom ViraCortes JSON")
    parser.add_argument("--template-mode", choices=("existing", "horizontal", "fill"))
    parser.add_argument("--gameplay-split", action="store_true", help="stack the webcam above gameplay without a frame")
    parser.add_argument("--webcam-crop", type=_webcam_crop, metavar="X,Y,W,H",
                        help="override opening-frame webcam detection with a crop in source pixels")
    parser.add_argument("--preview-at", type=float, metavar="SECONDS",
                        help="also write a PNG at this time within each clip")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clipper", description="Long video / live VOD -> vertical viral clips")
    parser.add_argument("--out", default="out", help="output root folder (default: out)")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", aliases=["gameplay"], help="download, select and render clips",
                           description="Use 'clipper gameplay VIDEO' for the Codex gameplay profile.")
    run_p.add_argument("source", help="URL (YouTube, Twitch, Kick...) or local video file")
    run_p.add_argument("--max-clips", type=int, default=8)
    run_p.add_argument("--min-dur", type=float, default=20.0)
    run_p.add_argument("--max-dur", type=float, default=75.0)
    run_p.add_argument("--lang", default=None, help="force transcript language, e.g. pt")
    run_p.add_argument("--chat", default=None, help="chat log: yt-dlp live_chat.json, TwitchDownloader JSON or list of seconds")
    run_p.add_argument("--batch", default=None, help="name for a new batch of clips; existing batches are preserved")
    run_p.add_argument("--exclude-existing", action="store_true", help="avoid moments already rendered as clips")
    run_p.add_argument("--no-visual-review", action="store_true",
                       help="skip frame review for Codex gameplay and select by transcript")
    run_p.add_argument("--prefer-speaker", default=None, help="participant to prioritize using supplied speaker timestamps")
    run_p.add_argument("--speaker-turns", type=Path, default=None, help="JSON with start/end/speaker annotations; required with --prefer-speaker")
    run_p.add_argument("--speaker-mode", choices=("prefer", "only"), default="prefer", help="prefer this participant or allow only their speech")
    run_p.add_argument("--speaker-weight", type=float, default=0.25, help="speaker share weight in ranking, from 0 to 1 (default: 0.25)")
    run_p.add_argument("--content-type", choices=("auto", "podcast", "gameplay", "anime", "mass"), default="auto",
                       help="layout/editorial mode; gameplay, anime and mass preserve the full source frame")
    run_p.add_argument("--llm-window-seconds", type=int, default=None, metavar="SECONDS",
                       help="transcript span per LLM call (default: 900; use 300 on a notebook)")
    run_p.add_argument("--llm-output-tokens", type=int, default=None, metavar="TOKENS",
                       help="maximum output tokens per LLM call (default: 1200)")
    run_p.add_argument("--notebook", action="store_true",
                       help="local low-memory profile: Ollama qwen3:4b, 5-min windows and 8k context")
    run_p.add_argument("--campaign", type=Path, default=None, metavar="BRIEF.json",
                       help="campaign brief; selects and reviews clips against its requirements")
    run_p.add_argument("--review-llm", default=None, metavar="PROVIDER[:MODEL][,...]",
                       help="independent LLM chain for the campaign review (defaults to --llm)")
    run_p.add_argument("--review-mode", choices=("strict", "warn"), default="strict",
                       help="strict renders only approved clips; warn keeps all and records the audit")
    run_p.add_argument("--llm", default=None, metavar="PROVIDER[:MODEL][,...]",
                       help="override LLM_PROVIDER; a comma list is a fallback chain, "
                            f"e.g. anthropic,openai:gpt-5-mini,ollama ({', '.join(PROVIDERS)})")
    _add_render_options(run_p)

    render_p = sub.add_parser("render", help="re-render clips from an edited highlights.json")
    render_p.add_argument("workdir", help="folder created by `run` (out/<video>)")
    render_p.add_argument("--batch", default=None, help="name for this new render batch; preserves previous batches")
    render_p.add_argument("--content-type", choices=("auto", "podcast", "gameplay", "anime", "mass"), default="auto")
    _add_render_options(render_p)
    render_p.add_argument("--limit", type=int, help="render only the highest-ranked N clips")
    render_p.add_argument("--preview-only", action="store_true", help="preview the top clip without exporting MP4")
    overlay_p = sub.add_parser("overlay", help="generate the transparent ViraCortes PNG")
    overlay_p.add_argument("output", type=Path)
    overlay_p.add_argument("--template-config", type=Path)
    overlay_p.add_argument("--template-mode", choices=("existing", "horizontal", "fill"))
    metrics_p = sub.add_parser("metrics", help="report attention by segment without an LLM or rendering")
    metrics_p.add_argument("workdir", help="existing video output folder")
    metrics_p.add_argument("--source", default=None, help="refresh public replay data from this video URL")
    diarize_p = sub.add_parser("diarize", help="separate voices locally and map names from reference intervals")
    diarize_p.add_argument("workdir", type=Path)
    diarize_p.add_argument("--segmentation-model", type=Path, required=True)
    diarize_p.add_argument("--embedding-model", type=Path, required=True)
    diarize_p.add_argument("--num-speakers", type=int, default=4)
    diarize_p.add_argument("--threshold", type=float, default=0.25)
    diarize_p.add_argument("--window-shift", type=float, default=0.5)
    diarize_p.add_argument("--chunk-seconds", type=int, default=300)
    diarize_p.add_argument("--anchor", action="append", default=[], metavar="NAME=START:END")
    diarize_p.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "gameplay":
        args.command = "run"
        args.content_type = "gameplay"
        args.llm = args.llm or "codex"
    if args.command == "diarize":
        from clipper.diarize import run as diarize
        try:
            diarize(args.workdir, args.segmentation_model, args.embedding_model, args.num_speakers,
                    args.threshold, args.window_shift, args.anchor, args.output, args.chunk_seconds)
        except ValueError as exc:
            print(f"[error] {exc}", file=sys.stderr, flush=True)
            return 2
        return 0
    if args.command == "metrics":
        report = analyze(Path(args.workdir), args.source)
        print(f"[done] metrics in {report} and {report.with_suffix('.md')}", flush=True)
        return 0
    if args.command != "overlay" and args.webcam_crop is not None and not args.gameplay_split:
        parser.error("--webcam-crop requires --gameplay-split")
    if args.command != "overlay" and args.gameplay_split:
        if args.content_type not in {"auto", "gameplay"}:
            parser.error("--gameplay-split requires gameplay")
        args.content_type = "gameplay"
    try:
        brand = None
        if args.command == "overlay" or (args.viracortes and not args.gameplay_split):
            brand = load_branding(args.template_config, args.template_mode)
        if args.command == "overlay":
            from clipper.brand_graphics import write_overlay

            if args.output.suffix.lower() != ".png":
                parser.error("overlay output must have a .png extension")
            print(write_overlay(brand, args.output), flush=True)
            return 0
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.preview_at is not None and (not math.isfinite(args.preview_at) or args.preview_at < 0):
        parser.error("--preview-at must be finite and nonnegative")
    if args.command == "render" and args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.command == "run" and (args.max_clips < 1 or args.min_dur <= 0 or args.max_dur < args.min_dur):
        build_parser().error("max-clips must be positive and 0 < min-dur <= max-dur")
    if args.command == "run":
        if bool(args.prefer_speaker) != bool(args.speaker_turns):
            build_parser().error("use --prefer-speaker together with --speaker-turns; the current transcript has no speaker identification")
        if args.speaker_mode == "only" and not args.prefer_speaker:
            build_parser().error("speaker-mode only requires --prefer-speaker and --speaker-turns")
        if args.speaker_turns and not args.speaker_turns.is_file():
            build_parser().error(f"speaker timeline not found: {args.speaker_turns}")
        if args.campaign and not args.campaign.is_file():
            build_parser().error(f"campaign brief not found: {args.campaign}")
        if not math.isfinite(args.speaker_weight) or not 0 <= args.speaker_weight <= 1:
            build_parser().error("speaker-weight must be between 0 and 1")
    settings = Settings.from_env()
    if args.command == "run":
        if args.llm_window_seconds is not None and not 120 <= args.llm_window_seconds <= 1800:
            build_parser().error("llm-window-seconds must be between 120 and 1800")
        if args.llm_output_tokens is not None and not 128 <= args.llm_output_tokens <= 4096:
            build_parser().error("llm-output-tokens must be between 128 and 4096")
        if args.notebook and args.llm is None:
            args.llm = "ollama:qwen3:4b"
        settings = settings.with_llm(args.llm)
        if args.notebook:
            settings = settings.with_limits(
                args.llm_output_tokens or 768,
                8192,
            )
        elif args.llm_output_tokens is not None:
            settings = settings.with_limits(args.llm_output_tokens)
        unknown = [provider for provider, _ in settings.llm_chain if provider not in PROVIDERS]
        if unknown:
            build_parser().error(f"unknown LLM provider {unknown[0]!r}; use one of: {', '.join(PROVIDERS)}")
        options = Options(max_clips=args.max_clips, min_dur=args.min_dur, max_dur=args.max_dur,
                          language=args.lang, captions=not args.no_captions,
                          batch=args.batch, exclude_existing=args.exclude_existing,
                          prefer_speaker=args.prefer_speaker, speaker_turns=args.speaker_turns,
                          speaker_mode=args.speaker_mode, speaker_weight=args.speaker_weight,
                          content_type=args.content_type,
                          llm_window_seconds=args.llm_window_seconds or (300.0 if args.notebook else 900.0),
                          campaign_file=args.campaign, review_llm=args.review_llm, review_mode=args.review_mode,
                          visual_review=not args.no_visual_review, brand=brand, preview_at=args.preview_at,
                          gameplay_split=args.gameplay_split, webcam_crop=args.webcam_crop)
        try:
            workdir = run(args.source, Path(args.out), settings, options, args.chat)
        except CodexResponseRequired as exc:
            print(f"\nCodex | {len(exc.requests)} pedido(s) aguardando resposta", flush=True)
            for request in exc.requests:
                print(request, flush=True)
            print("Responda com $clipper-cortes e repita o mesmo comando para continuar.", flush=True)
            return 3
        except FileNotFoundError as exc:
            print(f"[error] video not found: {exc}. Pass a URL (https://...) or the path of a video file.",
                  file=sys.stderr, flush=True)
            return 2
        except ProvidersExhausted as exc:
            print(f"[error] {exc}", file=sys.stderr, flush=True)
            return 2
        except ValueError as exc:
            print(f"[error] {exc}", file=sys.stderr, flush=True)
            return 2
    else:
        try:
            workdir = rerender(Path(args.workdir), settings,
                               Options(captions=not args.no_captions, content_type=args.content_type,
                                       batch=args.batch, brand=brand, preview_only=args.preview_only,
                                       preview_at=args.preview_at, render_limit=args.limit,
                                       gameplay_split=args.gameplay_split, webcam_crop=args.webcam_crop))
        except ValueError as exc:
            print(f"[error] {exc}", file=sys.stderr, flush=True)
            return 2
        if args.preview_only:
            return 0
    print(f"[done] clips in {workdir / 'clips'}", flush=True)
    return 0
