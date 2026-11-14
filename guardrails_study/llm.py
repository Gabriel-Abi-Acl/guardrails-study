"""Cliente de LLM agnóstico de provedor.

O caminho de teste é a fila do ScriptedLLM. SDKs reais só entram na primeira
chamada ao vivo, para a suíte offline não depender de pacote nem de chave.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


class TokenUsage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)


class LLMResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    text: str
    usage: TokenUsage | None = None
    model: str | None = None


class LLMClient(Protocol):
    def complete(self, messages: list[dict[str, str]], *, model: str | None = None) -> LLMResponse:
        """Devolve o texto do modelo e, se o provedor informar, a contagem de tokens."""


class ScriptedOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    usage: TokenUsage | None = None


class ScriptedLLM:
    """Fila determinística de saídas. Não recicla e não inventa texto quando acaba."""

    def __init__(self, outputs: list[str | ScriptedOutput]) -> None:
        self._outputs = list(outputs)
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages: list[dict[str, str]], *, model: str | None = None) -> LLMResponse:
        self.calls.append([{key: value for key, value in item.items()} for item in messages])
        if not self._outputs:
            raise RuntimeError("scripted LLM queue is empty")
        item = self._outputs.pop(0)
        if isinstance(item, str):
            return LLMResponse(text=item, usage=None, model=model)
        return LLMResponse(text=item.text, usage=item.usage, model=model)


# Temperatura fixa nos adaptadores ao vivo para o experimento não variar amostragem.
LIVE_TEMPERATURE = 0


def split_system_messages(
    messages: list[dict[str, str]],
) -> tuple[str | None, list[dict[str, str]]]:
    """Anthropic recebe system fora da lista. Os outros papéis permanecem na ordem."""
    systems = [item["content"] for item in messages if item["role"] == "system"]
    rest = [item for item in messages if item["role"] != "system"]
    if not systems:
        return None, rest
    return "\n\n".join(systems), rest


def usage_from_counts(prompt: object, completion: object) -> TokenUsage | None:
    # bool passa em isinstance(..., int). Sem os dois ints, o uso fica ausente — não zera.
    if type(prompt) is not int or type(completion) is not int:
        return None
    return TokenUsage(prompt_tokens=prompt, completion_tokens=completion)


def usage_from_openai_payload(usage: object | None) -> TokenUsage | None:
    if usage is None:
        return None
    return usage_from_counts(
        getattr(usage, "prompt_tokens", None),
        getattr(usage, "completion_tokens", None),
    )


def usage_from_anthropic_payload(usage: object | None) -> TokenUsage | None:
    if usage is None:
        return None
    return usage_from_counts(
        getattr(usage, "input_tokens", None),
        getattr(usage, "output_tokens", None),
    )


def text_from_anthropic_content(content: object) -> str:
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def _import_openai_class() -> object:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "The openai package is not installed. Install with: uv sync --extra live"
        ) from exc
    return OpenAI


def _import_anthropic_class() -> object:
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise ImportError(
            "The anthropic package is not installed. Install with: uv sync --extra live"
        ) from exc
    return Anthropic


class OpenAIClient:
    """Adaptador fino. O import de openai ocorre no primeiro complete, não no import do pacote."""

    def __init__(
        self, api_key: str, model: str, *, max_output_tokens: int = 1024, loader=None
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._loader = loader or _import_openai_class
        self._client = None

    def __repr__(self) -> str:
        return f"OpenAIClient(model={self._model!r})"

    def complete(self, messages: list[dict[str, str]], *, model: str | None = None) -> LLMResponse:
        if self._client is None:
            sdk_cls = self._loader()
            self._client = sdk_cls(api_key=self._api_key)
        chosen = model or self._model
        response = self._client.chat.completions.create(
            model=chosen,
            messages=list(messages),
            temperature=LIVE_TEMPERATURE,
        )
        content = response.choices[0].message.content
        text = content if isinstance(content, str) else ""
        return LLMResponse(text=text, usage=usage_from_openai_payload(response.usage), model=chosen)


class AnthropicClient:
    """Adaptador fino. O import de anthropic ocorre no primeiro complete."""

    def __init__(
        self, api_key: str, model: str, *, max_output_tokens: int = 1024, loader=None
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._loader = loader or _import_anthropic_class
        self._client = None

    def __repr__(self) -> str:
        return f"AnthropicClient(model={self._model!r})"

    def complete(self, messages: list[dict[str, str]], *, model: str | None = None) -> LLMResponse:
        if self._client is None:
            sdk_cls = self._loader()
            self._client = sdk_cls(api_key=self._api_key)
        chosen = model or self._model
        system, rest = split_system_messages(messages)
        kwargs: dict[str, object] = {
            "model": chosen,
            "max_tokens": self._max_output_tokens,
            "messages": rest,
            "temperature": LIVE_TEMPERATURE,
        }
        if system is not None:
            kwargs["system"] = system
        response = self._client.messages.create(**kwargs)
        return LLMResponse(
            text=text_from_anthropic_content(response.content),
            usage=usage_from_anthropic_payload(response.usage),
            model=chosen,
        )
