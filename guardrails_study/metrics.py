"""Métricas contra o gabarito, calculadas fora das guardas.

`valid` continua sendo o aceite da guarda. Erro de transporte não entra no
denominador e não é tratado como saída errada.
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict

from guardrails_study.dataset import EVAL_CATEGORIES

SCHEMA_VERSION = 1

# Ordem estável de leitura. O conjunto tem de bater com o runner.
STRATEGY_ORDER: tuple[str, ...] = ("json_schema", "pydantic", "rules", "second_llm")


class ClassFlags(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    escaped_invalid: bool
    false_rejection: bool
    budget_exhausted: bool


class Observation(BaseModel):
    """Uma execução já encerrada. A agregação não chama LLM nem guarda."""

    model_config = ConfigDict(extra="forbid")
    strategy: str
    category: str
    status: Literal["ok", "error"]
    valid: bool | None = None
    matches_ground_truth: bool | None = None
    attempts: int | None = None
    max_attempts: int | None = None
    latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None


class Distribution(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mean: float
    p50: float
    p90: float


class MetricBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scored_count: int
    error_count: int
    acceptance_rate: float | None
    escaped_invalid_rate: float | None
    false_rejection_rate: float | None
    budget_exhaustion_rate: float | None
    attempts: Distribution | None
    latency_ms: Distribution | None
    prompt_tokens: int | None
    completion_tokens: int | None
    cost_usd: float | None


class StrategyBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")
    overall: MetricBlock
    by_category: dict[str, MetricBlock]


class Summary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: int
    dataset: str
    backend: str
    scenario: str | None
    strategies: list[str]
    max_attempts: int
    record_count: int
    note: str
    overall: MetricBlock
    by_strategy: dict[str, StrategyBlock]
    by_category: dict[str, MetricBlock]


def classify(
    *,
    valid: bool,
    matches_ground_truth: bool,
    attempts: int,
    max_attempts: int,
) -> ClassFlags:
    """Separa aceite da guarda e acordo com o gabarito.

    Esgotar o orçamento exige recusa na última tentativa. Aceitar no último
    passo não conta como exaustão.
    """
    return ClassFlags(
        escaped_invalid=valid and not matches_ground_truth,
        false_rejection=(not valid) and matches_ground_truth,
        budget_exhausted=(not valid) and attempts >= max_attempts,
    )


def output_matches_ground_truth(raw: str, ground_truth: Mapping[str, object]) -> bool:
    """JSON parseado igual ao gabarito. Texto ilegível não é o gabarito.

    Bool não equivale a 0 ou 1. Int e float com o mesmo valor equivalem,
    porque o JSON do dataset não distingue 20 de 20.0. Ordem de chave não conta.
    """
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return False
    return _same_json(parsed, ground_truth)


def nearest_rank(values: Sequence[float], percentile: float) -> float:
    """Percentil nearest-rank: posição ceil(p/100 * n) na lista ordenada, base 1."""
    if not values:
        raise ValueError("nearest_rank requires at least one value")
    if percentile <= 0 or percentile > 100:
        raise ValueError("percentile must be in (0, 100]")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile / 100 * len(ordered)))
    return float(ordered[rank - 1])


def aggregate(observations: Sequence[Observation]) -> MetricBlock:
    """Agrega só linhas ok. Linha error sai do denominador e não vira zero."""
    errors = [item for item in observations if item.status == "error"]
    scored = [item for item in observations if item.status == "ok"]
    for item in scored:
        _require_scored(item)
    if not scored:
        return _empty_block(error_count=len(errors))
    flags = [
        classify(
            valid=item.valid is True,
            matches_ground_truth=item.matches_ground_truth is True,
            attempts=item.attempts or 0,
            max_attempts=item.max_attempts or 0,
        )
        for item in scored
    ]
    count = len(scored)
    return MetricBlock(
        scored_count=count,
        error_count=len(errors),
        acceptance_rate=sum(item.valid is True for item in scored) / count,
        escaped_invalid_rate=sum(item.escaped_invalid for item in flags) / count,
        false_rejection_rate=sum(item.false_rejection for item in flags) / count,
        budget_exhaustion_rate=sum(item.budget_exhausted for item in flags) / count,
        attempts=_distribution([float(item.attempts or 0) for item in scored]),
        latency_ms=_distribution([float(item.latency_ms or 0) for item in scored]),
        prompt_tokens=_sum_ints([item.prompt_tokens for item in scored]),
        completion_tokens=_sum_ints([item.completion_tokens for item in scored]),
        cost_usd=_sum_floats([item.cost_usd for item in scored]),
    )


def build_summary(
    observations: Sequence[Observation],
    *,
    dataset: str,
    backend: str,
    scenario: str | None,
    strategies: Sequence[str],
    max_attempts: int,
    note: str,
) -> Summary:
    """Quebra o mesmo conjunto por estratégia e por categoria, sem reexecutar."""
    by_strategy: dict[str, StrategyBlock] = {}
    for strategy in _strategy_keys(observations):
        group = [item for item in observations if item.strategy == strategy]
        by_category = {
            category: aggregate([item for item in group if item.category == category])
            for category in _category_keys(group)
        }
        by_strategy[strategy] = StrategyBlock(
            overall=aggregate(group),
            by_category=by_category,
        )
    by_category = {
        category: aggregate([item for item in observations if item.category == category])
        for category in _category_keys(observations)
    }
    return Summary(
        schema_version=SCHEMA_VERSION,
        dataset=dataset,
        backend=backend,
        scenario=scenario,
        strategies=list(strategies),
        max_attempts=max_attempts,
        record_count=len(observations),
        note=note,
        overall=aggregate(observations),
        by_strategy=by_strategy,
        by_category=by_category,
    )


def _same_json(left: object, right: object) -> bool:
    if type(left) is bool or type(right) is bool:
        return type(left) is bool and type(right) is bool and left == right
    if type(left) in (int, float) and type(right) in (int, float):
        return float(left) == float(right)
    if isinstance(left, dict) and isinstance(right, Mapping):
        if set(left) != set(right):
            return False
        return all(_same_json(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return False
        return all(_same_json(item, other) for item, other in zip(left, right, strict=True))
    return left == right


def _distribution(values: list[float]) -> Distribution:
    return Distribution(
        mean=float(statistics.fmean(values)),
        p50=nearest_rank(values, 50),
        p90=nearest_rank(values, 90),
    )


def _empty_block(*, error_count: int) -> MetricBlock:
    return MetricBlock(
        scored_count=0,
        error_count=error_count,
        acceptance_rate=None,
        escaped_invalid_rate=None,
        false_rejection_rate=None,
        budget_exhaustion_rate=None,
        attempts=None,
        latency_ms=None,
        prompt_tokens=None,
        completion_tokens=None,
        cost_usd=None,
    )


def _require_scored(item: Observation) -> None:
    missing = [
        name
        for name in ("valid", "matches_ground_truth", "attempts", "max_attempts", "latency_ms")
        if getattr(item, name) is None
    ]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"scored observation is missing {joined}")


def _sum_ints(values: list[int | None]) -> int | None:
    # Null em qualquer linha anula a soma. Tratar como zero inventaria uso.
    if any(value is None for value in values):
        return None
    return sum(value or 0 for value in values)


def _sum_floats(values: list[float | None]) -> float | None:
    if any(value is None for value in values):
        return None
    return float(math.fsum(value or 0.0 for value in values))


def _strategy_keys(observations: Sequence[Observation]) -> list[str]:
    present = {item.strategy for item in observations}
    ordered = [name for name in STRATEGY_ORDER if name in present]
    ordered.extend(name for name in _first_seen(observations, "strategy") if name not in ordered)
    return ordered


def _category_keys(observations: Sequence[Observation]) -> list[str]:
    present = {item.category for item in observations}
    ordered = [name for name in EVAL_CATEGORIES if name in present]
    ordered.extend(name for name in _first_seen(observations, "category") if name not in ordered)
    return ordered


def _first_seen(observations: Sequence[Observation], field: str) -> list[str]:
    seen: list[str] = []
    for item in observations:
        value = getattr(item, field)
        if value not in seen:
            seen.append(value)
    return seen
