"""Configuração só por ambiente. Segredo não entra em repr nem em mensagem de erro."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from guardrails_study.llm import AnthropicClient, LLMClient, OpenAIClient, ScriptedLLM

_PROVIDERS = frozenset({"fake", "openai", "anthropic"})


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    llm_provider: Literal["fake", "openai", "anthropic"] = "fake"
    llm_model: str = ""
    llm_api_key: str = Field(default="", repr=False)
    checker_provider: Literal["fake", "openai", "anthropic"] = "fake"
    checker_model: str = ""
    checker_api_key: str = Field(default="", repr=False)
    max_attempts: int = Field(default=3, ge=1)
    llm_max_output_tokens: int = Field(default=1024, ge=1)
    price_prompt_per_million: float | None = Field(default=None, ge=0)
    price_completion_per_million: float | None = Field(default=None, ge=0)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        source = os.environ if env is None else env
        return cls(
            llm_provider=_provider(source, "LLM_PROVIDER"),
            llm_model=_optional_text(source, "LLM_MODEL"),
            llm_api_key=_optional_text(source, "LLM_API_KEY"),
            checker_provider=_provider(source, "CHECKER_PROVIDER"),
            checker_model=_optional_text(source, "CHECKER_MODEL"),
            checker_api_key=_optional_text(source, "CHECKER_API_KEY"),
            max_attempts=_positive_int(source, "MAX_ATTEMPTS", 3),
            llm_max_output_tokens=_positive_int(source, "LLM_MAX_OUTPUT_TOKENS", 1024),
            price_prompt_per_million=_optional_price(source, "PRICE_PROMPT_PER_MILLION"),
            price_completion_per_million=_optional_price(source, "PRICE_COMPLETION_PER_MILLION"),
        )


def _optional_text(env: Mapping[str, str], name: str) -> str:
    raw = env.get(name)
    if raw is None:
        return ""
    return raw.strip()


def _provider(env: Mapping[str, str], name: str) -> str:
    raw = _optional_text(env, name) or "fake"
    if raw not in _PROVIDERS:
        raise ValueError(f"{name} must be one of: fake, openai, anthropic")
    return raw


def _positive_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


def _optional_price(env: Mapping[str, str], name: str) -> float | None:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _role_fields(settings: Settings, role: str) -> tuple[str, str, str, str, str]:
    if role == "generator":
        return (
            settings.llm_provider,
            settings.llm_model,
            settings.llm_api_key,
            "LLM_MODEL",
            "LLM_API_KEY",
        )
    if role == "checker":
        return (
            settings.checker_provider,
            settings.checker_model,
            settings.checker_api_key,
            "CHECKER_MODEL",
            "CHECKER_API_KEY",
        )
    raise ValueError(f"unknown role: {role}")


def build_llm(
    settings: Settings,
    *,
    role: str = "generator",
    scripted: ScriptedLLM | None = None,
    loader=None,
) -> LLMClient:
    """Monta o cliente. Provedor fake exige a fila; provedor ao vivo exige modelo e chave."""
    provider, model, api_key, model_var, key_var = _role_fields(settings, role)
    if provider == "fake":
        if scripted is None:
            raise ValueError(f"{role} provider fake requires a ScriptedLLM")
        return scripted
    if not model:
        raise ValueError(f"{model_var} is required when provider is {provider}")
    if not api_key:
        raise ValueError(f"{key_var} is required when provider is {provider}")
    if provider == "openai":
        client_cls = OpenAIClient
    elif provider == "anthropic":
        client_cls = AnthropicClient
    else:
        raise ValueError(f"unknown provider: {provider}")
    kwargs = {
        "api_key": api_key,
        "model": model,
        "max_output_tokens": settings.llm_max_output_tokens,
    }
    if loader is not None:
        kwargs["loader"] = loader
    return client_cls(**kwargs)
