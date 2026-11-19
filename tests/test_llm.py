import json
from types import SimpleNamespace

import pytest

from clipper import llm
from clipper.cli import build_parser, main
from clipper.config import DEFAULT_MODELS, Settings

SCHEMA = {"type": "object"}


def test_with_llm_provider_only_keeps_default_model():
    s = Settings().with_llm("openai")
    assert s.llm_provider == "openai"
    assert s.model_for("openai") == DEFAULT_MODELS["openai"]


def test_with_llm_model_containing_colon_and_slash():
    assert Settings().with_llm("ollama:qwen3:8b").model_for("ollama") == "qwen3:8b"
    s = Settings().with_llm("OpenRouter:meta-llama/llama-4-maverick")
    assert s.llm_provider == "openrouter"
    assert s.model_for("openrouter") == "meta-llama/llama-4-maverick"


def test_with_llm_none_is_noop():
    s = Settings(llm_provider="gemini")
    assert s.with_llm(None) is s


def test_env_overrides_models(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "Gemini")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-x")
    s = Settings.from_env()
    assert s.llm_provider == "gemini" and s.model_for("gemini") == "gemini-x"
    assert s.model_for("anthropic") == DEFAULT_MODELS["anthropic"]


def test_build_unknown_provider():
    with pytest.raises(ValueError, match="unknown LLM provider"):
        llm.build(Settings(llm_provider="nope"))


def test_cli_rejects_unknown_provider(capsys):
    with pytest.raises(SystemExit):
        main(["run", "video.mp4", "--llm", "nope"])
    assert "unknown LLM provider" in capsys.readouterr().err


def test_cli_parses_llm_flag():
    args = build_parser().parse_args(["run", "x.mp4", "--llm", "openai:gpt-5"])
    assert args.llm == "openai:gpt-5"


def test_cli_parses_notebook_profile():
    args = build_parser().parse_args(["run", "x.mp4", "--notebook"])
    assert args.notebook is True and args.llm_window_seconds is None


def test_cli_parses_campaign_review_options():
    args = build_parser().parse_args(["run", "x.mp4", "--campaign", "brief.json",
                                      "--review-llm", "gemini", "--review-mode", "warn"])
    assert str(args.campaign) == "brief.json"
    assert args.review_llm == "gemini" and args.review_mode == "warn"


def test_gameplay_profile_keeps_selection_and_visual_models_independent(tmp_path, monkeypatch):
    captured = {}

    def fake_run(source, out, settings, options, chat):
        captured.update(settings=settings, options=options)
        return tmp_path

    monkeypatch.setattr("clipper.cli.run", fake_run)
    assert main(["gameplay", "video.mp4", "--llm", "gemini:test-model",
                 "--visual-review-llm", "codex:reviewer"]) == 0
    assert captured["settings"].llm_provider == "gemini"
    assert captured["options"].content_type == "gameplay"
    assert captured["options"].visual_review is True
    assert captured["options"].visual_review_llm == "codex:reviewer"


def test_visual_review_rejects_unsupported_provider_before_running(monkeypatch):
    monkeypatch.setattr("clipper.cli.run", lambda *args: pytest.fail("run must not start"))
    with pytest.raises(SystemExit):
        main(["gameplay", "video.mp4", "--llm", "gemini:test", "--visual-review-llm", "gemini:test"])


def test_openai_compatible_sends_strict_schema_and_parses(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            msg = SimpleNamespace(content=json.dumps({"highlights": []}), refusal=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    class FakeOpenAI:
        def __init__(self, api_key=None, base_url=None):
            captured["base_url"] = base_url
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setattr(llm, "_require", lambda module, extra: SimpleNamespace(OpenAI=FakeOpenAI))
    call = llm.build(Settings().with_llm("openrouter:some/model"))
    assert call("sys", "user", SCHEMA) == {"highlights": []}
    assert captured["base_url"] == llm.OPENROUTER_URL
    assert captured["model"] == "some/model"
    assert captured["response_format"]["json_schema"]["strict"] is True
    assert captured["messages"][0] == {"role": "system", "content": "sys"}


def test_openai_refusal_raises(monkeypatch):
    class FakeOpenAI:
        def __init__(self, **_):
            msg = SimpleNamespace(content=None, refusal="no")
            create = lambda **kw: SimpleNamespace(choices=[SimpleNamespace(message=msg)])
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    monkeypatch.setattr(llm, "_require", lambda module, extra: SimpleNamespace(OpenAI=FakeOpenAI))
    with pytest.raises(llm.LLMRefusal):
        llm.build(Settings(llm_provider="openai"))("s", "u", SCHEMA)


def test_gemini_passes_json_schema(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self):
            def generate_content(**kwargs):
                captured.update(kwargs)
                return SimpleNamespace(text='{"highlights": [1]}')
            self.models = SimpleNamespace(generate_content=generate_content)

    monkeypatch.setattr(llm, "_require", lambda module, extra: SimpleNamespace(Client=FakeClient))
    result = llm.build(Settings().with_llm("gemini:gemini-z"))("sys", "user", SCHEMA)
    assert result == {"highlights": [1]}
    assert captured["model"] == "gemini-z"
    assert captured["config"]["response_json_schema"] is SCHEMA
    assert captured["config"]["system_instruction"] == "sys"


def test_missing_sdk_gives_install_hint(monkeypatch):
    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "openai":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(llm.ProvidersExhausted, match=r"MissingDependency.*\.\[openai\]"):
        llm.build(Settings(llm_provider="openai"))("s", "u", SCHEMA)


def test_with_llm_comma_list_builds_fallback_chain():
    s = Settings().with_llm("anthropic, openai:gpt-5 ,ollama:qwen3:8b,Gemini")
    assert s.llm_chain == (
        ("anthropic", DEFAULT_MODELS["anthropic"]),
        ("openai", "gpt-5"),
        ("ollama", "qwen3:8b"),
        ("gemini", DEFAULT_MODELS["gemini"]),
    )


def test_single_llm_flag_drops_env_fallbacks(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic,openai")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-x")
    s = Settings.from_env()
    assert s.llm_chain == (("anthropic", DEFAULT_MODELS["anthropic"]), ("openai", "gpt-x"))
    assert s.with_llm("gemini").llm_chain == (("gemini", DEFAULT_MODELS["gemini"]),)


def test_cli_rejects_unknown_provider_in_chain(capsys):
    with pytest.raises(SystemExit):
        main(["run", "video.mp4", "--llm", "openai,nope"])
    assert "'nope'" in capsys.readouterr().err


def _failing(message):
    def call(system, user, schema):
        raise RuntimeError(message)
    return call


def test_chain_switches_provider_and_stays_on_it():
    calls = []

    def good(system, user, schema):
        calls.append(user)
        return {"ok": user}

    def broken_factory():
        raise llm.MissingDependency("no sdk")

    chain = llm.LLMChain([("a:1", lambda: _failing("insufficient_quota")),
                          ("b:2", broken_factory), ("c:3", lambda: good)])
    assert chain.label == "a:1"
    assert chain("s", "u1", SCHEMA) == {"ok": "u1"}
    assert chain.label == "c:3"
    assert chain("s", "u2", SCHEMA) == {"ok": "u2"}
    assert calls == ["u1", "u2"]
    assert "insufficient_quota" in chain.failures[0] and "no sdk" in chain.failures[1]


def test_chain_refusal_does_not_drop_provider():
    def refuse(system, user, schema):
        raise llm.LLMRefusal("nope")

    chain = llm.LLMChain([("a:1", lambda: refuse), ("b:2", lambda: _failing("x"))])
    with pytest.raises(llm.LLMRefusal):
        chain("s", "u", SCHEMA)
    assert chain.label == "a:1"


def test_chain_exhausted_lists_every_error():
    chain = llm.LLMChain([("a:1", lambda: _failing("rate limit")), ("b:2", lambda: _failing("credit"))])
    with pytest.raises(llm.ProvidersExhausted, match="a:1.*rate limit.*b:2.*credit"):
        chain("s", "u", SCHEMA)


@pytest.mark.parametrize("finish_reason,content,expected_error", [
    ("stop", '{"highlights": []}', None),
    ("length", '{"highlights": []}', "token limit"),
    ("stop", "   ", "empty JSON"),
])
def test_deepseek_json_mode_and_incomplete_output(monkeypatch, finish_reason, content, expected_error):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

            def create(**request):
                captured.update(request)
                return SimpleNamespace(choices=[SimpleNamespace(
                    finish_reason=finish_reason, message=SimpleNamespace(content=content, refusal=None))])

            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(llm, "_require", lambda module, extra: SimpleNamespace(OpenAI=FakeOpenAI))
    call = llm.build(Settings().with_llm("deepseek"))
    if expected_error:
        with pytest.raises(llm.ProvidersExhausted, match=expected_error):
            call("system", "user", SCHEMA)
    else:
        assert call("system", "user", SCHEMA) == {"highlights": []}
    assert captured["base_url"] == llm.DEEPSEEK_URL
    assert captured["response_format"] == {"type": "json_object"}
    assert "JSON Schema" in captured["messages"][0]["content"]
    assert captured["max_tokens"] == 1200
    assert captured["model"] == DEFAULT_MODELS["deepseek"]


def test_missing_deepseek_key_uses_next_model(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setitem(llm.PROVIDERS, "gemini", lambda s, m: lambda *args: {"highlights": []})
    chain = llm.build(Settings().with_llm("deepseek,gemini:backup"))
    assert chain("s", "u", SCHEMA) == {"highlights": []}
    assert chain.label == "gemini:backup"
    assert "DEEPSEEK_API_KEY" in chain.failures[0]


def test_switch_models_within_same_provider(monkeypatch):
    monkeypatch.setitem(llm.PROVIDERS, "deepseek", lambda s, m:
                        _failing("429 quota") if m == "first" else lambda *args: {"ok": True})
    chain = llm.build(Settings().with_llm("deepseek:first,deepseek:second"))
    assert chain("s", "u", SCHEMA) == {"ok": True}
    assert chain.label == "deepseek:second"
