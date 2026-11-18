import argparse
import sys
import math
from pathlib import Path

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

    run_p = sub.add_parser("run", help="download/transcribe/detect/render")
    run_p.add_argument("source", help="URL (YouTube, Twitch, Kick...) or local video file")
    run_p.add_argument("--max-clips", type=int, default=8)
    run_p.add_argument("--min-dur", type=float, default=20.0)
    run_p.add_argument("--max-dur", type=float, default=75.0)
    run_p.add_argument("--lang", default=None, help="force transcript language, e.g. pt")
    run_p.add_argument("--chat", default=None, help="chat log: yt-dlp live_chat.json, TwitchDownloader JSON or list of seconds")
    run_p.add_argument("--batch", default=None, help="name for a new batch of clips; existing batches are preserved")
    run_p.add_argument("--exclude-existing", action="store_true", help="avoid moments already rendered as clips")
    run_p.add_argument("--prefer-speaker", default=None, help="participant to prioritize using supplied speaker timestamps")
    run_p.add_argument("--speaker-turns", type=Path, default=None, help="JSON with start/end/speaker annotations; required with --prefer-speaker")
    run_p.add_argument("--speaker-mode", choices=("prefer", "only"), default="prefer", help="prefer this participant or allow only their speech")
    run_p.add_argument("--speaker-weight", type=float, default=0.25, help="speaker share weight in ranking, from 0 to 1 (default: 0.25)")
    run_p.add_argument("--content-type", choices=("auto", "podcast", "gameplay"), default="auto",
                       help="layout: auto detects game titles; gameplay preserves the full game screen")
    run_p.add_argument("--llm", default=None, metavar="PROVIDER[:MODEL][,...]",
                       help="override LLM_PROVIDER; a comma list is a fallback chain, "
                            f"e.g. anthropic,openai:gpt-5-mini,ollama ({', '.join(PROVIDERS)})")
    _add_render_options(run_p)

    render_p = sub.add_parser("render", help="re-render clips from an edited highlights.json")
    render_p.add_argument("workdir", help="folder created by `run` (out/<video>)")
    render_p.add_argument("--batch", default=None, help="name for this new render batch; preserves previous batches")
    render_p.add_argument("--content-type", choices=("auto", "podcast", "gameplay"), default="auto")
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
    args = build_parser().parse_args(argv)
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
        if not math.isfinite(args.speaker_weight) or not 0 <= args.speaker_weight <= 1:
            build_parser().error("speaker-weight must be between 0 and 1")
    settings = Settings.from_env()
    if args.command == "run":
        settings = settings.with_llm(args.llm)
        unknown = [provider for provider, _ in settings.llm_chain if provider not in PROVIDERS]
        if unknown:
            build_parser().error(f"unknown LLM provider {unknown[0]!r}; use one of: {', '.join(PROVIDERS)}")
        options = Options(max_clips=args.max_clips, min_dur=args.min_dur, max_dur=args.max_dur,
                          language=args.lang, captions=not args.no_captions,
                          batch=args.batch, exclude_existing=args.exclude_existing,
                          prefer_speaker=args.prefer_speaker, speaker_turns=args.speaker_turns,
                          speaker_mode=args.speaker_mode, speaker_weight=args.speaker_weight,
                          content_type=args.content_type)
        try:
            workdir = run(args.source, Path(args.out), settings, options, args.chat)
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
    print(f"[done] clips in {workdir / 'clips'}", flush=True)
    return 0
