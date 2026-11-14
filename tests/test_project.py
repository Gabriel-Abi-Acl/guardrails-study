"""Trava de repositório: CI pinado, docs em pt-BR e nenhum segredo versionado."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_ci_pins_setup_uv_tag():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "astral-sh/setup-uv@v10.3.3" in workflow
    assert "uv run ruff check ." in workflow
    assert "uv run pytest" in workflow


def test_env_example_is_the_only_env_file_and_has_no_secret():
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "LLM_PROVIDER=fake" in example
    assert "PRICE_PROMPT_PER_MILLION=" in example
    assert not (ROOT / ".env").exists()
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in gitignore.splitlines()
    for token in ("sk-", "sk-ant-", "api_key="):
        assert token not in example


@pytest.mark.parametrize(
    "heading",
    [
        "## Objetivo",
        "## Hipótese",
        "## Metodologia planejada",
        "## Dataset de avaliação",
        "## Como rodar",
        "## Avaliação offline",
        "## Roadmap",
    ],
)
def test_readme_has_the_planned_sections(heading):
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert heading in readme
    assert "PR2" in readme
    assert "PR3" in readme
    assert "PR4" in readme
    assert "não contém resultado experimental" in readme
