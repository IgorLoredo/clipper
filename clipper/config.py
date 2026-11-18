import os
from dataclasses import dataclass, field, replace
from pathlib import Path

DEFAULT_MODELS = {
    "anthropic": "claude-opus-5-5",
    "openai": "gpt-5-mini",
    "gemini": "gemini-3.6-flash",
    "deepseek": "deepseek-flash",
    "openrouter": "google/gemini-2.5-flash",
    "ollama": "qwen3:8b",
}


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if value and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class Settings:
    llm_provider: str = "anthropic"
    llm_fallbacks: tuple[tuple[str, str], ...] = ()
    models: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_MODELS))
    ollama_url: str = "http://localhost:11434"
    whisper_model: str = ""
    whisper_device: str = ""
    face_model: str = ""

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ.get
        models = {name: env(f"{name.upper()}_MODEL") or default for name, default in DEFAULT_MODELS.items()}
        settings = cls(
            models=models,
            ollama_url=env("OLLAMA_URL", "http://localhost:11434").rstrip("/"),
            whisper_model=env("WHISPER_MODEL", ""),
            whisper_device=env("WHISPER_DEVICE", ""),
            face_model=env("FACE_MODEL", ""),
        )
        return settings.with_llm(env("LLM_PROVIDER", "anthropic"))

    def model_for(self, provider: str) -> str:
        return self.models.get(provider) or DEFAULT_MODELS.get(provider, "")

    @property
    def llm_chain(self) -> tuple[tuple[str, str], ...]:
        return ((self.llm_provider, self.model_for(self.llm_provider)),) + self.llm_fallbacks

    def with_llm(self, spec: str | None) -> "Settings":
        items = [item.strip() for item in (spec or "").split(",") if item.strip()]
        if not items:
            return self
        chain = []
        for item in items:
            provider, _, model = item.partition(":")
            provider = provider.strip().lower()
            chain.append((provider, model.strip() or self.model_for(provider)))
        provider, model = chain[0]
        models = dict(self.models)
        models[provider] = model
        return replace(self, llm_provider=provider, models=models, llm_fallbacks=tuple(chain[1:]))
