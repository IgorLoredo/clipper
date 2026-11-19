import json
import os
import urllib.request
from functools import partial
from typing import Callable

from clipper.codex_exchange import CodexExchange, CodexResponseRequired
from clipper.config import Settings

JsonLLM = Callable[[str, str, dict], dict]

OPENROUTER_URL = "https://openrouter.ai/api/v1"
DEEPSEEK_URL = "https://api.deepseek.com"


class LLMRefusal(RuntimeError):
    pass


class MissingDependency(RuntimeError):
    pass


class ProvidersExhausted(RuntimeError):
    pass


class LLMChain:
    def __init__(self, entries: list[tuple[str, Callable[[], JsonLLM]]]):
        self._entries = list(entries)
        self._call: JsonLLM | None = None
        self.failures: list[str] = []

    @property
    def label(self) -> str:
        return self._entries[0][0] if self._entries else ""

    def __call__(self, system: str, user: str, schema: dict) -> dict:
        while self._entries:
            label, factory = self._entries[0]
            try:
                if self._call is None:
                    self._call = factory()
                    print(f"[llm] using {label}", flush=True)
                return self._call(system, user, schema)
            except (LLMRefusal, CodexResponseRequired):
                raise
            except Exception as exc:
                self._drop(label, exc)
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


def _anthropic(model: str, max_output_tokens: int) -> JsonLLM:
    anthropic = _require("anthropic", "anthropic")
    client = anthropic.Anthropic()

    def call(system: str, user: str, schema: dict) -> dict:
        response = client.beta.messages.create(
            model=model,
            max_tokens=max_output_tokens,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={
                "effort": "medium",
                "format": {"type": "json_schema", "schema": schema},
            },
        )
        if response.stop_reason == "refusal":
            raise LLMRefusal(str(getattr(response, "stop_details", "")))
        text = next(b.text for b in response.content if b.type == "text")
        return json.loads(text)

    return call


def _openai_compatible(model: str, max_output_tokens: int, api_key: str | None = None,
                       base_url: str | None = None, strict_schema: bool = True) -> JsonLLM:
    openai = _require("openai", "openai")
    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def call(system: str, user: str, schema: dict) -> dict:
        if strict_schema:
            response_format = {"type": "json_schema", "json_schema": {
                "name": "highlights", "schema": schema, "strict": True,
            }}
            options = {"max_tokens": max_output_tokens}
        else:
            response_format = {"type": "json_object"}
            system += "\nReturn only JSON matching this JSON Schema:\n" + json.dumps(schema)
            system += '\nExample JSON structure: {"highlights": []}'
            options = {"max_tokens": max_output_tokens}
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format=response_format,
            **options,
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


def _openai(model: str, max_output_tokens: int) -> JsonLLM:
    return _openai_compatible(model, max_output_tokens)


def _openrouter(model: str, max_output_tokens: int) -> JsonLLM:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    return _openai_compatible(model, max_output_tokens, api_key=key, base_url=OPENROUTER_URL)


def _deepseek(model: str, max_output_tokens: int) -> JsonLLM:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    return _openai_compatible(model, max_output_tokens, api_key=key, base_url=DEEPSEEK_URL,
                              strict_schema=False)


def _gemini(model: str, max_output_tokens: int) -> JsonLLM:
    genai = _require("google.genai", "gemini")
    client = genai.Client()

    def call(system: str, user: str, schema: dict) -> dict:
        response = client.models.generate_content(
            model=model,
            contents=user,
            config={
                "system_instruction": system,
                "response_mime_type": "application/json",
                "response_json_schema": schema,
                "temperature": 0.3,
                "max_output_tokens": max_output_tokens,
            },
        )
        if not response.text:
            raise LLMRefusal(str(getattr(response, "prompt_feedback", "") or "empty response"))
        return json.loads(response.text)

    return call


def _ollama(model: str, url: str, max_output_tokens: int, context_tokens: int) -> JsonLLM:
    def call(system: str, user: str, schema: dict) -> dict:
        body = json.dumps({
            "model": model,
            "stream": False,
            "format": schema,
            "think": False,
            "options": {"temperature": 0.3, "num_ctx": context_tokens,
                        "num_predict": max_output_tokens},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }).encode("utf-8")
        request = urllib.request.Request(f"{url}/api/chat", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=1800) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if payload.get("done_reason") == "length":
            raise LLMRefusal("Ollama response reached its output limit")
        content = payload.get("message", {}).get("content", "").strip()
        if not content:
            raise LLMRefusal("Ollama returned an empty response")
        return json.loads(content)

    return call


PROVIDERS: dict[str, Callable[[Settings, str], JsonLLM]] = {
    "anthropic": lambda s, m: _anthropic(m, s.llm_max_output_tokens),
    "openai": lambda s, m: _openai(m, s.llm_max_output_tokens),
    "gemini": lambda s, m: _gemini(m, s.llm_max_output_tokens),
    "deepseek": lambda s, m: _deepseek(m, s.llm_max_output_tokens),
    "openrouter": lambda s, m: _openrouter(m, s.llm_max_output_tokens),
    "ollama": lambda s, m: _ollama(m, s.ollama_url, s.llm_max_output_tokens,
                                     s.ollama_context_tokens),
    "codex": lambda s, m: CodexExchange(s.codex_directory, m),
}


def build(settings: Settings) -> LLMChain:
    entries = []
    for provider, model in settings.llm_chain:
        factory = PROVIDERS.get(provider)
        if factory is None:
            raise ValueError(f"unknown LLM provider {provider!r}; use one of: {', '.join(PROVIDERS)}")
        entries.append((f"{provider}:{model}", partial(factory, settings, model)))
    print(f"[llm] chain: {' -> '.join(label for label, _ in entries)}", flush=True)
    return LLMChain(entries)
