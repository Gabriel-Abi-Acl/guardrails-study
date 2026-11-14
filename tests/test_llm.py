"""Fake offline e adaptadores ao vivo sem rede. O SDK só carrega na primeira chamada."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from types import SimpleNamespace

import pytest

from guardrails_study.config import Settings, build_llm
from guardrails_study.llm import (
    AnthropicClient,
    OpenAIClient,
    ScriptedLLM,
    split_system_messages,
    text_from_anthropic_content,
    usage_from_anthropic_payload,
    usage_from_openai_payload,
)


def test_importing_the_package_does_not_import_live_sdks():
    importlib.import_module("guardrails_study")
    importlib.import_module("guardrails_study.llm")
    assert "openai" not in sys.modules
    assert "anthropic" not in sys.modules


def test_scripted_llm_does_not_recycle_the_queue():
    client = ScriptedLLM(["one"])
    assert client.complete([{"role": "user", "content": "ping"}]).text == "one"
    with pytest.raises(RuntimeError, match="queue is empty"):
        client.complete([{"role": "user", "content": "ping"}])
    assert len(client.calls) == 2


def test_usage_parsers_skip_missing_or_bool_counts():
    assert usage_from_openai_payload(None) is None
    assert (
        usage_from_openai_payload(SimpleNamespace(prompt_tokens=True, completion_tokens=1)) is None
    )
    parsed = usage_from_openai_payload(SimpleNamespace(prompt_tokens=2, completion_tokens=3))
    assert parsed is not None
    assert parsed.prompt_tokens == 2
    assert (
        usage_from_anthropic_payload(SimpleNamespace(input_tokens=4, output_tokens=5)) is not None
    )
    assert (
        text_from_anthropic_content(
            [SimpleNamespace(text="a"), SimpleNamespace(type="tool"), SimpleNamespace(text="b")]
        )
        == "ab"
    )


def test_split_system_messages_keeps_conversation_order():
    system, rest = split_system_messages(
        [
            {"role": "system", "content": "one"},
            {"role": "user", "content": "ask"},
            {"role": "system", "content": "two"},
        ]
    )
    assert system == "one\n\ntwo"
    assert rest == [{"role": "user", "content": "ask"}]


def test_openai_client_loads_sdk_on_first_complete_and_hides_the_key():
    loads: list[str] = []

    class FakeSDK:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.chat = self
            self.completions = self
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4),
            )

    def loader():
        loads.append("load")
        return FakeSDK

    client = OpenAIClient(api_key="super-secret-value", model="gpt-test", loader=loader)
    assert loads == []
    assert "super-secret-value" not in repr(client)
    first = client.complete([{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}])
    second = client.complete([{"role": "user", "content": "again"}])
    assert loads == ["load"]
    assert first.text == "ok"
    assert first.usage is not None
    assert first.usage.completion_tokens == 4
    assert second.usage is not None
    assert client._client.kwargs["temperature"] == 0
    assert client._client.kwargs["model"] == "gpt-test"
    assert "super-secret-value" not in first.text


def test_anthropic_client_splits_system_and_reads_input_tokens():
    class FakeSDK:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.messages = self
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                content=[SimpleNamespace(text='{"ok": true}')],
                usage=SimpleNamespace(input_tokens=5, output_tokens=6),
            )

    client = AnthropicClient(
        api_key="anthropic-secret", model="claude-test", loader=lambda: FakeSDK
    )
    assert "anthropic-secret" not in repr(client)
    response = client.complete(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "u"}]
    )
    assert response.text == '{"ok": true}'
    assert response.usage is not None
    assert response.usage.prompt_tokens == 5
    assert client._client.kwargs["system"] == "sys"
    assert client._client.kwargs["messages"] == [{"role": "user", "content": "u"}]
    assert client._client.kwargs["max_tokens"] == 1024
    assert client._client.kwargs["temperature"] == 0


def test_default_loaders_report_missing_sdks():
    if importlib.util.find_spec("openai") is not None:
        pytest.skip("openai is installed")
    if importlib.util.find_spec("anthropic") is not None:
        pytest.skip("anthropic is installed")
    client = OpenAIClient(api_key="test-key-openai", model="m")
    with pytest.raises(ImportError, match="live"):
        client.complete([{"role": "user", "content": "hi"}])
    other = AnthropicClient(api_key="test-key-anthropic", model="m")
    with pytest.raises(ImportError, match="live"):
        other.complete([{"role": "user", "content": "hi"}])


def test_build_llm_fake_and_live_credentials():
    scripted = ScriptedLLM(["{}"])
    settings = Settings()
    assert build_llm(settings, scripted=scripted) is scripted
    with pytest.raises(ValueError, match="ScriptedLLM"):
        build_llm(settings)

    live = Settings(llm_provider="openai", llm_model="", llm_api_key="")
    with pytest.raises(ValueError, match="LLM_MODEL"):
        build_llm(live)
    live = Settings(llm_provider="openai", llm_model="m", llm_api_key="")
    with pytest.raises(ValueError, match="LLM_API_KEY is required") as exc:
        build_llm(live)
    assert "sk-" not in str(exc.value)

    def loader() -> object:
        raise RuntimeError("loader must not run at build time")

    built = build_llm(
        Settings(llm_provider="anthropic", llm_model="m", llm_api_key="secret-value"),
        loader=loader,
    )
    assert isinstance(built, AnthropicClient)
    assert "secret-value" not in repr(built)
