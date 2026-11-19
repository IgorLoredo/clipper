import argparse
import sys
import math
from pathlib import Path

from clipper.codex_exchange import CodexResponseRequired, complete_status, write_status
from clipper.config import Settings, load_dotenv
from clipper.llm import PROVIDERS, ProvidersExhausted
from clipper.pipeline import Options, rerender, run
from clipper.attention import analyze


def _add_render_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-captions", action="store_true", help="do not burn captions")


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
    visual = run_p.add_mutually_exclusive_group()
    visual.add_argument("--visual-review", dest="visual_review", action="store_true", default=None,
                        help="review gameplay candidates using local frame grids")
    visual.add_argument("--no-visual-review", dest="visual_review", action="store_false",
                        help="select gameplay only from transcript and existing signals")
    run_p.add_argument("--visual-review-llm", default="codex", metavar="codex[:MODEL]",
                       help="independent visual reviewer (default: codex)")
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
    command_args = list(argv) if argv is not None else sys.argv[1:]
    args = build_parser().parse_args(argv)
    gameplay_profile = args.command == "gameplay"
    if gameplay_profile:
        args.command = "run"
        args.content_type = "gameplay"
        if args.visual_review is None:
            args.visual_review = True
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
        if args.visual_review:
            visual_settings = settings.with_llm(args.visual_review_llm)
            if visual_settings.llm_provider != "codex" or visual_settings.llm_fallbacks:
                build_parser().error("visual-review-llm currently accepts only codex[:model]")
        options = Options(max_clips=args.max_clips, min_dur=args.min_dur, max_dur=args.max_dur,
                          language=args.lang, captions=not args.no_captions,
                          batch=args.batch, exclude_existing=args.exclude_existing,
                          prefer_speaker=args.prefer_speaker, speaker_turns=args.speaker_turns,
                          speaker_mode=args.speaker_mode, speaker_weight=args.speaker_weight,
                          content_type=args.content_type,
                          llm_window_seconds=args.llm_window_seconds or (300.0 if args.notebook else 900.0),
                          campaign_file=args.campaign, review_llm=args.review_llm, review_mode=args.review_mode,
                          visual_review=bool(args.visual_review), visual_review_llm=args.visual_review_llm)
        try:
            workdir = run(args.source, Path(args.out), settings, options, args.chat)
        except CodexResponseRequired as exc:
            status = write_status(exc.requests, ["clipper", *command_args], exc.stage)
            print(f"\nCodex | {len(exc.requests)} pedido(s) aguardando resposta", flush=True)
            for request in exc.requests:
                print(request, flush=True)
            print(f"Status: {status}", flush=True)
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
        workdir = rerender(Path(args.workdir), settings,
                           Options(captions=not args.no_captions, content_type=args.content_type,
                                   batch=args.batch))
    complete_status(workdir / "codex-llm", ["clipper", *command_args])
    print(f"[done] clips in {workdir / 'clips'}", flush=True)
    return 0
