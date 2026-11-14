"""Ambiente e fábrica de cliente. Chave não aparece em repr nem em erro."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from guardrails_study.config import Settings


def test_from_env_defaults_to_fake_without_prices():
    settings = Settings.from_env({})
    assert settings.llm_provider == "fake"
    assert settings.checker_provider == "fake"
    assert settings.max_attempts == 3
    assert settings.price_prompt_per_million is None
    assert settings.price_completion_per_million is None
    assert settings.llm_api_key == ""


def test_from_env_parses_prices_and_attempts():
    settings = Settings.from_env(
        {
            "LLM_PROVIDER": "openai",
            "LLM_MODEL": "gpt-test",
            "LLM_API_KEY": "secret-value",
            "CHECKER_PROVIDER": "anthropic",
            "CHECKER_MODEL": "claude-test",
            "CHECKER_API_KEY": "other-secret",
            "MAX_ATTEMPTS": "4",
            "PRICE_PROMPT_PER_MILLION": "1.5",
            "PRICE_COMPLETION_PER_MILLION": "2",
        }
    )
    assert settings.llm_provider == "openai"
    assert settings.max_attempts == 4
    assert settings.price_prompt_per_million == pytest.approx(1.5)
    assert "secret-value" not in repr(settings)
    assert "other-secret" not in repr(settings)


@pytest.mark.parametrize(
    ("env", "match"),
    [
        ({"LLM_PROVIDER": "local-llama"}, "LLM_PROVIDER"),
        ({"MAX_ATTEMPTS": "0"}, "MAX_ATTEMPTS"),
        ({"MAX_ATTEMPTS": "nope"}, "MAX_ATTEMPTS"),
        ({"PRICE_PROMPT_PER_MILLION": "-1"}, "PRICE_PROMPT_PER_MILLION"),
        ({"PRICE_COMPLETION_PER_MILLION": "free"}, "PRICE_COMPLETION_PER_MILLION"),
    ],
)
def test_from_env_rejects_invalid_values(env, match):
    with pytest.raises(ValueError, match=match):
        Settings.from_env(env)


def test_blank_api_key_is_stripped():
    settings = Settings.from_env({"LLM_API_KEY": "   "})
    assert settings.llm_api_key == ""


def test_settings_reject_non_positive_attempt_budget():
    with pytest.raises(ValidationError):
        Settings(max_attempts=0)
