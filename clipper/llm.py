import base64
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request
from functools import partial
from typing import Callable

from clipper.codex_exchange import CodexExchange, CodexResponseRequired
from clipper.config import Settings
from clipper.usage import UsageTracker

JsonLLM = Callable[..., dict]

OPENROUTER_URL = "https://openrouter.ai/api/v1"
DEEPSEEK_URL = "https://api.deepseek.com"
_TRANSIENT_PATTERN = re.compile(
    r"\b(?:502|503|504)\b|temporarily unavailable|overloaded|connection reset|timed out|read timeout|try again later",
    re.IGNORECASE,
)


class LLMRefusal(RuntimeError):
    pass


class MissingDependency(RuntimeError):
    pass


class ProvidersExhausted(RuntimeError):
    pass


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, (MissingDependency, ValueError, KeyError, LLMRefusal, CodexResponseRequired)):
        return False
    if isinstance(exc, (TimeoutError, ConnectionError, urllib.error.URLError)):
        return True
    return bool(_TRANSIENT_PATTERN.search(str(exc)))


def _schema_example(schema: dict) -> str:
    props = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(props, dict) or not props:
        return '{"highlights": []}'
    example = {}
    for key, spec in props.items():
        kind = spec.get("type") if isinstance(spec, dict) else None
        if kind == "array":
            example[key] = []
        elif kind == "object":
            example[key] = {}
        elif kind == "boolean":
            example[key] = False
        elif kind in {"integer", "number"}:
            example[key] = 0
        else:
            example[key] = ""
    return json.dumps(example, ensure_ascii=False)


def _load_images(images: list[dict] | None, base_dir: Path | None = None) -> list[tuple[str, bytes, str]]:
    if not images:
        return []
    loaded: list[tuple[str, bytes, str]] = []
    for item in images:
        raw_path = Path(item["file"])
        path = raw_path if raw_path.is_file() else (base_dir / raw_path if base_dir else raw_path)
        if not path.is_file():
            continue
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        data = path.read_bytes()
        label = f"Candidate index {item['index']}" if "index" in item else path.name
        loaded.append((mime, data, label))
    return loaded


class LLMChain:
    def __init__(self, entries: list[tuple[str, Callable[[], JsonLLM]]],
                 tracker: UsageTracker | None = None, max_retries: int = 2,
                 retry_delay: float = 0.5):
        self._entries = list(entries)
        self._call: JsonLLM | None = None
        self.failures: list[str] = []
        self.usage = tracker or UsageTracker()
        self.max_retries = max(0, max_retries)
        self.retry_delay = max(0.0, retry_delay)

    @property
    def label(self) -> str:
        return self._entries[0][0] if self._entries else ""

    def __call__(self, system: str, user: str, schema: dict, *,
                 images: list[dict] | None = None) -> dict:
        while self._entries:
            label, factory = self._entries[0]
            attempts = 0
            while True:
                try:
                    if self._call is None:
                        self._call = factory()
                        print(f"[llm] using {label}", flush=True)
                    if images is not None:
                        return self._call(system, user, schema, images=images)
                    return self._call(system, user, schema)
                except (LLMRefusal, CodexResponseRequired):
                    raise
                except Exception as exc:
                    if attempts < self.max_retries and _is_transient(exc):
                        attempts += 1
                        wait = self.retry_delay * (2 ** (attempts - 1))
                        print(f"[llm] transient error on {label} ({type(exc).__name__}); "
                              f"retrying {attempts}/{self.max_retries} in {wait:.1f}s", flush=True)
                        if wait > 0:
                            time.sleep(wait)
                        continue
                    self._drop(label, exc)
                    break
        raise ProvidersExhausted(" | ".join(self.failures) or "no LLM configured")

    def _drop(self, label: str, exc: Exception) -> None:
        reason = f"{label}: {type(exc).__name__}: {str(exc)[:300]}"
        self.failures.append(reason)
        self._entries.pop(0)
        self._call = None
        following = f"switching to {self.label}" if self._entries else "no LLM left"
        print(f"[llm] {reason} -> {following}", flush=True)


def _require(module: str, extra: str):
    try:
        return __import__(module, fromlist=["_"])
    except ImportError as exc:
        raise MissingDependency(f"provider needs `{module}`: pip install -e \".[{extra}]\"") from exc


def _anthropic(model: str, max_output_tokens: int, tracker: UsageTracker | None = None,
               base_dir: Path | None = None) -> JsonLLM:
    anthropic = _require("anthropic", "anthropic")
    client = anthropic.Anthropic()

    def call(system: str, user: str, schema: dict, *, images: list[dict] | None = None) -> dict:
        loaded = _load_images(images, base_dir)
        if loaded:
            content: object = [{"type": "text", "text": user}]
            for mime, data, label in loaded:
                content.append({"type": "text", "text": label})
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": mime, "data": base64.b64encode(data).decode("ascii")},
                })
        else:
            content = user
        cached_system: object = (
            [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
            if system else system
        )
        response = client.beta.messages.create(
            model=model,
            max_tokens=max_output_tokens,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=cached_system,
            messages=[{"role": "user", "content": content}],
            output_config={
                "effort": "medium",
                "format": {"type": "json_schema", "schema": schema},
            },
        )
        usage = getattr(response, "usage", None)
        if tracker and usage is not None:
            tracker.record(
                "anthropic", model,
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
                cached_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            )
        if response.stop_reason == "refusal":
            raise LLMRefusal(str(getattr(response, "stop_details", "")))
        text = next(b.text for b in response.content if b.type == "text")
        return json.loads(text)

    return call


def _openai_compatible(model: str, max_output_tokens: int, api_key: str | None = None,
                       base_url: str | None = None, strict_schema: bool = True,
                       provider_name: str = "openai", tracker: UsageTracker | None = None,
                       base_dir: Path | None = None) -> JsonLLM:
    openai = _require("openai", "openai")
    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def call(system: str, user: str, schema: dict, *, images: list[dict] | None = None) -> dict:
        sys_prompt = system
        if strict_schema:
            response_format = {"type": "json_schema", "json_schema": {
                "name": "highlights", "schema": schema, "strict": True,
            }}
            options = {"max_tokens": max_output_tokens}
        else:
            response_format = {"type": "json_object"}
            sys_prompt += "\nReturn only JSON matching this JSON Schema:\n" + json.dumps(schema)
            sys_prompt += "\nExample JSON structure: " + _schema_example(schema)
            options = {"max_tokens": max_output_tokens}
        loaded = _load_images(images, base_dir)
        if loaded:
            user_content: object = [{"type": "text", "text": user}]
            for mime, data, label in loaded:
                b64 = base64.b64encode(data).decode("ascii")
                user_content.append({"type": "text", "text": label})
                user_content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        else:
            user_content = user
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_content}],
            response_format=response_format,
            **options,
        )
        usage = getattr(response, "usage", None)
        if tracker and usage is not None:
            details = getattr(usage, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", 0) if details is not None else 0
            tracker.record(
                provider_name, model,
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                cached_tokens=cached or 0,
                total_tokens=getattr(usage, "total_tokens", None),
            )
        if getattr(response.choices[0], "finish_reason", None) == "length":
            raise RuntimeError("LLM output token limit reached; response was truncated")
        message = response.choices[0].message
        if getattr(message, "refusal", None):
            raise LLMRefusal(message.refusal)
        if not message.content or not message.content.strip():
            raise RuntimeError("LLM returned an empty JSON response")
        payload = json.loads(message.content)
        if not isinstance(payload, dict):
            raise ValueError("LLM response must be a JSON object")
        return payload

    return call


def _openai(model: str, max_output_tokens: int, tracker: UsageTracker | None = None,
            base_dir: Path | None = None) -> JsonLLM:
    return _openai_compatible(model, max_output_tokens, provider_name="openai", tracker=tracker, base_dir=base_dir)


def _openrouter(model: str, max_output_tokens: int, tracker: UsageTracker | None = None,
                base_dir: Path | None = None) -> JsonLLM:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    return _openai_compatible(model, max_output_tokens, api_key=key, base_url=OPENROUTER_URL,
                              provider_name="openrouter", tracker=tracker, base_dir=base_dir)


def _deepseek(model: str, max_output_tokens: int, tracker: UsageTracker | None = None,
              base_dir: Path | None = None) -> JsonLLM:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    return _openai_compatible(model, max_output_tokens, api_key=key, base_url=DEEPSEEK_URL,
                              strict_schema=False, provider_name="deepseek", tracker=tracker, base_dir=base_dir)


def _gemini(model: str, max_output_tokens: int, tracker: UsageTracker | None = None,
            base_dir: Path | None = None) -> JsonLLM:
    genai = _require("google.genai", "gemini")
    client = genai.Client()

    def call(system: str, user: str, schema: dict, *, images: list[dict] | None = None) -> dict:
        loaded = _load_images(images, base_dir)
        if loaded:
            part_cls = getattr(getattr(genai, "types", None), "Part", None)
            contents: object = [user]
            for mime, data, label in loaded:
                contents.append(label)
                if part_cls is not None and hasattr(part_cls, "from_bytes"):
                    contents.append(part_cls.from_bytes(data=data, mime_type=mime))
                else:
                    contents.append({"inline_data": {"mime_type": mime, "data": base64.b64encode(data).decode("ascii")}})
        else:
            contents = user
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config={
                "system_instruction": system,
                "response_mime_type": "application/json",
                "response_json_schema": schema,
                "temperature": 0.3,
                "max_output_tokens": max_output_tokens,
            },
        )
        meta = getattr(response, "usage_metadata", None)
        if tracker and meta is not None:
            tracker.record(
                "gemini", model,
                input_tokens=getattr(meta, "prompt_token_count", 0) or 0,
                output_tokens=getattr(meta, "candidates_token_count", 0) or 0,
                cached_tokens=getattr(meta, "cached_content_token_count", 0) or 0,
                total_tokens=getattr(meta, "total_token_count", None),
            )
        if not response.text:
            raise LLMRefusal(str(getattr(response, "prompt_feedback", "") or "empty response"))
        return json.loads(response.text)

    return call


def _ollama(model: str, url: str, max_output_tokens: int, context_tokens: int,
            tracker: UsageTracker | None = None, base_dir: Path | None = None) -> JsonLLM:
    def call(system: str, user: str, schema: dict, *, images: list[dict] | None = None) -> dict:
        user_msg: dict[str, object] = {"role": "user", "content": user}
        loaded = _load_images(images, base_dir)
        if loaded:
            user_msg["images"] = [base64.b64encode(data).decode("ascii") for _, data, _ in loaded]
        body = json.dumps({
            "model": model,
            "stream": False,
            "format": schema,
            "think": False,
            "options": {"temperature": 0.3, "num_ctx": context_tokens,
                        "num_predict": max_output_tokens},
            "messages": [
                {"role": "system", "content": system},
                user_msg,
            ],
        }).encode("utf-8")
        request = urllib.request.Request(f"{url}/api/chat", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=1800) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if tracker:
            in_tok = payload.get("prompt_eval_count") or 0
            out_tok = payload.get("eval_count") or 0
            if in_tok or out_tok:
                tracker.record("ollama", model, input_tokens=in_tok, output_tokens=out_tok)
        if payload.get("done_reason") == "length":
            raise LLMRefusal("Ollama response reached its output limit")
        content = payload.get("message", {}).get("content", "").strip()
        if not content:
            raise LLMRefusal("Ollama returned an empty response")
        return json.loads(content)

    return call


def _make_provider(provider: str, settings: Settings, model: str, tracker: UsageTracker) -> JsonLLM:
    base_dir = settings.codex_directory
    if provider == "anthropic":
        return _anthropic(model, settings.llm_max_output_tokens, tracker=tracker, base_dir=base_dir)
    if provider == "openai":
        return _openai(model, settings.llm_max_output_tokens, tracker=tracker, base_dir=base_dir)
    if provider == "gemini":
        return _gemini(model, settings.llm_max_output_tokens, tracker=tracker, base_dir=base_dir)
    if provider == "deepseek":
        return _deepseek(model, settings.llm_max_output_tokens, tracker=tracker, base_dir=base_dir)
    if provider == "openrouter":
        return _openrouter(model, settings.llm_max_output_tokens, tracker=tracker, base_dir=base_dir)
    if provider == "ollama":
        return _ollama(model, settings.ollama_url, settings.llm_max_output_tokens,
                       settings.ollama_context_tokens, tracker=tracker, base_dir=base_dir)
    if provider == "codex":
        return CodexExchange(settings.codex_directory, model)
    raise ValueError(f"unknown LLM provider {provider!r}; use one of: {', '.join(PROVIDERS)}")


PROVIDERS: dict[str, Callable[[Settings, str], JsonLLM]] = {
    "anthropic": lambda s, m: _anthropic(m, s.llm_max_output_tokens, base_dir=s.codex_directory),
    "openai": lambda s, m: _openai(m, s.llm_max_output_tokens, base_dir=s.codex_directory),
    "gemini": lambda s, m: _gemini(m, s.llm_max_output_tokens, base_dir=s.codex_directory),
    "deepseek": lambda s, m: _deepseek(m, s.llm_max_output_tokens, base_dir=s.codex_directory),
    "openrouter": lambda s, m: _openrouter(m, s.llm_max_output_tokens, base_dir=s.codex_directory),
    "ollama": lambda s, m: _ollama(m, s.ollama_url, s.llm_max_output_tokens,
                                     s.ollama_context_tokens, base_dir=s.codex_directory),
    "codex": lambda s, m: CodexExchange(s.codex_directory, m),
}


def build(settings: Settings, tracker: UsageTracker | None = None) -> LLMChain:
    tracker = tracker or UsageTracker()
    entries = []
    for provider, model in settings.llm_chain:
        factory = PROVIDERS.get(provider)
        if factory is None:
            raise ValueError(f"unknown LLM provider {provider!r}; use one of: {', '.join(PROVIDERS)}")
        if provider in _DEFAULT_FACTORIES and factory is _DEFAULT_FACTORIES[provider]:
            entries.append((f"{provider}:{model}", partial(_make_provider, provider, settings, model, tracker)))
        else:
            entries.append((f"{provider}:{model}", partial(factory, settings, model)))
    print(f"[llm] chain: {' -> '.join(label for label, _ in entries)}", flush=True)
    return LLMChain(entries, tracker=tracker)


_DEFAULT_FACTORIES = dict(PROVIDERS)
