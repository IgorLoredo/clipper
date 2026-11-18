"""Campaign briefs and an independent compliance review for generated clips."""

import hashlib
import json
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from clipper.llm import JsonLLM
from clipper.transcribe import Transcript

if TYPE_CHECKING:
    from clipper.detect import Highlight


REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "reviews": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "approved": {"type": "boolean"},
                    "score": {"type": "integer"},
                    "reasons": {"type": "array", "items": {"type": "string"}},
                    "suggested_title": {"type": "string"},
                    "caption": {"type": "string"},
                    "hashtags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["index", "approved", "score", "reasons", "suggested_title", "caption", "hashtags"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["reviews"],
    "additionalProperties": False,
}


def _text(value: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", value.casefold()) if not unicodedata.combining(c))


def _strings(data: dict, key: str) -> tuple[str, ...]:
    value = data.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"campaign {key} must be a list of non-empty strings")
    return tuple(item.strip() for item in value)


@dataclass(frozen=True)
class CampaignBrief:
    name: str
    objective: str
    audience: str = ""
    language: str = ""
    min_duration: float | None = None
    max_duration: float | None = None
    required_terms: tuple[str, ...] = ()
    forbidden_terms: tuple[str, ...] = ()
    required_hashtags: tuple[str, ...] = ()
    instructions: str = ""

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:12]

    def summary(self) -> str:
        rows = [f"Campaign: {self.name}", f"Objective: {self.objective}"]
        if self.audience:
            rows.append(f"Audience: {self.audience}")
        if self.language:
            rows.append(f"Required language: {self.language}")
        if self.min_duration is not None or self.max_duration is not None:
            rows.append(f"Duration: {self.min_duration or 0:g}-{self.max_duration or 'unbounded'} seconds")
        if self.required_terms:
            rows.append("Must include these spoken terms when naturally applicable: " + ", ".join(self.required_terms))
        if self.forbidden_terms:
            rows.append("Forbidden terms/topics: " + ", ".join(self.forbidden_terms))
        if self.required_hashtags:
            rows.append("Required hashtags: " + ", ".join(self.required_hashtags))
        if self.instructions:
            rows.append("Extra requirements: " + self.instructions)
        return "\n".join(rows)


def load(path: Path) -> CampaignBrief:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"campaign file is not valid JSON: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError("campaign file must contain a JSON object")
    name = str(data.get("name", "")).strip()
    objective = str(data.get("objective", "")).strip()
    if not name or not objective:
        raise ValueError("campaign requires non-empty name and objective")
    minimum = data.get("min_duration")
    maximum = data.get("max_duration")
    try:
        minimum = float(minimum) if minimum is not None else None
        maximum = float(maximum) if maximum is not None else None
    except (TypeError, ValueError) as exc:
        raise ValueError("campaign durations must be numbers") from exc
    if minimum is not None and minimum <= 0:
        raise ValueError("campaign min_duration must be positive")
    if maximum is not None and (maximum <= 0 or minimum is not None and maximum < minimum):
        raise ValueError("campaign max_duration must be at least min_duration")
    return CampaignBrief(
        name=name, objective=objective, audience=str(data.get("audience", "")).strip(),
        language=str(data.get("language", "")).strip(), min_duration=minimum, max_duration=maximum,
        required_terms=_strings(data, "required_terms"), forbidden_terms=_strings(data, "forbidden_terms"),
        required_hashtags=_strings(data, "required_hashtags"), instructions=str(data.get("instructions", "")).strip(),
    )


def deterministic_issues(brief: CampaignBrief, highlight: "Highlight", transcript: Transcript) -> list[str]:
    spoken = _text(" ".join(word.text for word in transcript.words_between(highlight.start, highlight.end)))
    issues = []
    if brief.min_duration is not None and highlight.duration < brief.min_duration:
        issues.append(f"duration is below the campaign minimum ({brief.min_duration:g}s)")
    if brief.max_duration is not None and highlight.duration > brief.max_duration:
        issues.append(f"duration is above the campaign maximum ({brief.max_duration:g}s)")
    issues.extend(f"missing required spoken term: {term}" for term in brief.required_terms if _text(term) not in spoken)
    issues.extend(f"contains forbidden term: {term}" for term in brief.forbidden_terms if _text(term) in spoken)
    return issues


def review(brief: CampaignBrief, highlights: list["Highlight"], transcript: Transcript,
           llm: JsonLLM, cache: Path) -> list[dict]:
    signature = [{"start": h.start, "end": h.end, "title": h.title, "score": h.score} for h in highlights]
    params = {"brief": brief.fingerprint, "clips": signature}
    if cache.is_file():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        if saved.get("params") == params:
            return saved["reviews"]
    candidates = []
    local_issues = []
    for index, highlight in enumerate(highlights):
        words = transcript.words_between(highlight.start, highlight.end)
        spoken = " ".join(word.text for word in words)[:3500]
        candidates.append({"index": index, "start": highlight.start, "end": highlight.end,
                           "title": highlight.title, "hook": highlight.hook, "transcript": spoken})
        local_issues.append(deterministic_issues(brief, highlight, transcript))
    system = """You are an independent campaign compliance reviewer. Evaluate each proposed short video against the brief.
Approve only clips that accurately fit the objective, audience, safety/tone and stated requirements. Do not invent spoken evidence.
Give concise actionable reasons. Suggested title and caption must be truthful and ready to publish; include required hashtags only when they fit."""
    user = "Campaign brief:\n" + brief.summary() + "\n\nCandidate clips:\n" + json.dumps(candidates, ensure_ascii=False)
    response = llm(system, user, REVIEW_SCHEMA)
    by_index = {item.get("index"): item for item in response.get("reviews", []) if isinstance(item, dict)}
    reviews = []
    for index, issues in enumerate(local_issues):
        item = by_index.get(index, {})
        approved = bool(item.get("approved", False)) and not issues
        reviews.append({
            "approved": approved,
            "score": max(0, min(100, int(item.get("score", 0)))),
            "reasons": issues + [str(reason) for reason in item.get("reasons", []) if str(reason).strip()],
            "suggested_title": str(item.get("suggested_title", "")).strip(),
            "caption": str(item.get("caption", "")).strip(),
            "hashtags": [str(tag).strip() for tag in item.get("hashtags", []) if str(tag).strip()],
            "deterministic_issues": issues,
        })
    cache.write_text(json.dumps({"params": params, "brief": asdict(brief), "reviews": reviews}, ensure_ascii=False, indent=2),
                     encoding="utf-8")
    return reviews
