"""Lightweight local token usage tracking across LLM providers."""

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
import threading


@dataclass(frozen=True)
class UsageEvent:
    timestamp: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int = 0
    total_tokens: int = 0
    stage: str = ""


class UsageTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: list[UsageEvent] = []

    @property
    def events(self) -> list[UsageEvent]:
        with self._lock:
            return list(self._events)

    def record(self, provider: str, model: str, input_tokens: int = 0, output_tokens: int = 0,
               cached_tokens: int = 0, total_tokens: int | None = None, stage: str = "") -> UsageEvent | None:
        in_tok = max(0, int(input_tokens or 0))
        out_tok = max(0, int(output_tokens or 0))
        cached_tok = max(0, int(cached_tokens or 0))
        total = max(0, int(total_tokens)) if total_tokens is not None else in_tok + out_tok
        if not any((in_tok, out_tok, cached_tok, total)):
            return None
        event = UsageEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            provider=provider,
            model=model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cached_tokens=cached_tok,
            total_tokens=total,
            stage=stage,
        )
        with self._lock:
            self._events.append(event)
        cache_note = f" cached={cached_tok:,}" if cached_tok else ""
        print(f"[usage] {provider}:{model} in={in_tok:,} out={out_tok:,}{cache_note} total={total:,}", flush=True)
        return event

    def summary(self) -> dict:
        with self._lock:
            events = list(self._events)
        return summarize_events(events)

    def save(self, workdir: Path | None = None, out_root: Path | None = None) -> None:
        with self._lock:
            events = list(self._events)
        if not events:
            return
        rows = [asdict(event) for event in events]
        if workdir is not None:
            path = workdir / "usage.json"
            existing: list[dict] = []
            if path.is_file():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    existing = list(data.get("events", []))
                except (OSError, ValueError):
                    existing = []
            merged = existing + rows
            payload = {
                "version": 1,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "summary": summarize_events([UsageEvent(**row) for row in merged]),
                "events": merged,
            }
            _atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2))
        if out_root is not None:
            out_root.mkdir(parents=True, exist_ok=True)
            log = out_root / "_usage.jsonl"
            with log.open("a", encoding="utf-8") as handle:
                for row in rows:
                    record = {**row, "workdir": workdir.name if workdir else ""}
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _atomic_write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(contents + "\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def summarize_events(events: list[UsageEvent]) -> dict:
    by_model: dict[str, dict[str, int]] = defaultdict(
        lambda: {"requests": 0, "input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "total_tokens": 0}
    )
    totals = {"requests": 0, "input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "total_tokens": 0}
    for event in events:
        key = f"{event.provider}:{event.model}"
        for target in (totals, by_model[key]):
            target["requests"] += 1
            target["input_tokens"] += event.input_tokens
            target["output_tokens"] += event.output_tokens
            target["cached_tokens"] += event.cached_tokens
            target["total_tokens"] += event.total_tokens
    return {"totals": totals, "by_model": dict(sorted(by_model.items()))}


def load_usage_report(out_root: Path, days: int = 7) -> dict:
    events: list[UsageEvent] = []
    log = out_root / "_usage.jsonl"
    if log.is_file():
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                events.append(UsageEvent(
                    timestamp=str(raw.get("timestamp", "")),
                    provider=str(raw.get("provider", "")),
                    model=str(raw.get("model", "")),
                    input_tokens=int(raw.get("input_tokens", 0)),
                    output_tokens=int(raw.get("output_tokens", 0)),
                    cached_tokens=int(raw.get("cached_tokens", 0)),
                    total_tokens=int(raw.get("total_tokens", 0)),
                    stage=str(raw.get("stage", "")),
                ))
            except (ValueError, TypeError):
                continue
    elif out_root.is_dir():
        for path in sorted(out_root.glob("*/usage.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                for raw in data.get("events", []):
                    events.append(UsageEvent(**raw))
            except (OSError, ValueError, TypeError):
                continue

    now = datetime.now(timezone.utc)
    today_prefix = now.date().isoformat()
    cutoff = now - timedelta(days=max(1, days))
    today_events: list[UsageEvent] = []
    window_events: list[UsageEvent] = []
    for event in events:
        if event.timestamp.startswith(today_prefix):
            today_events.append(event)
        try:
            ts = datetime.fromisoformat(event.timestamp.replace("Z", "+00:00"))
            if ts >= cutoff:
                window_events.append(event)
        except ValueError:
            window_events.append(event)

    return {
        "days": days,
        "today_utc": summarize_events(today_events),
        "window": summarize_events(window_events),
        "all_time": summarize_events(events),
    }


def format_usage_report(report: dict) -> str:
    lines = ["=== Clipper LLM Token Usage ==="]
    for label, section in (
        ("Today (UTC)", report["today_utc"]),
        (f"Last {report['days']} day(s)", report["window"]),
        ("All time", report["all_time"]),
    ):
        totals = section["totals"]
        lines.append(
            f"\n{label}: {totals['requests']} req(s) | "
            f"in={totals['input_tokens']:,} out={totals['output_tokens']:,} "
            f"cached={totals['cached_tokens']:,} total={totals['total_tokens']:,}"
        )
        for model, stats in section["by_model"].items():
            lines.append(
                f"  - {model}: {stats['requests']} req(s) | "
                f"in={stats['input_tokens']:,} out={stats['output_tokens']:,} "
                f"cached={stats['cached_tokens']:,} total={stats['total_tokens']:,}"
            )
    return "\n".join(lines)
