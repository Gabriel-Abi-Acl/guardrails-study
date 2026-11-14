"""Matemática das métricas em registros montados à mão. Não roda modelo nem guarda."""

from __future__ import annotations

import pytest

from guardrails_study.dataset import load_eval_tasks
from guardrails_study.metrics import (
    STRATEGY_ORDER,
    Observation,
    aggregate,
    build_summary,
    classify,
    nearest_rank,
    output_matches_ground_truth,
)
from guardrails_study.runner import STRATEGIES


def _ok(
    strategy: str,
    category: str,
    *,
    valid: bool,
    matches: bool,
    attempts: int,
    max_attempts: int,
    latency_ms: float,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    cost_usd: float | None = None,
) -> Observation:
    return Observation(
        strategy=strategy,
        category=category,
        status="ok",
        valid=valid,
        matches_ground_truth=matches,
        attempts=attempts,
        max_attempts=max_attempts,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost_usd,
    )


def test_strategy_order_matches_the_runner():
    assert set(STRATEGY_ORDER) == STRATEGIES
    assert STRATEGY_ORDER == ("json_schema", "pydantic", "rules", "second_llm")


def test_classify_separates_escape_false_rejection_and_exhaustion():
    escaped = classify(valid=True, matches_ground_truth=False, attempts=1, max_attempts=1)
    assert escaped.escaped_invalid is True
    assert escaped.false_rejection is False
    assert escaped.budget_exhausted is False

    rejected_label = classify(valid=False, matches_ground_truth=True, attempts=2, max_attempts=3)
    assert rejected_label.false_rejection is True
    assert rejected_label.budget_exhausted is False

    exhausted = classify(valid=False, matches_ground_truth=True, attempts=3, max_attempts=3)
    assert exhausted.false_rejection is True
    assert exhausted.budget_exhausted is True

    accepted_on_last = classify(valid=True, matches_ground_truth=True, attempts=3, max_attempts=3)
    assert accepted_on_last.budget_exhausted is False

    broken = classify(valid=False, matches_ground_truth=False, attempts=1, max_attempts=1)
    assert broken.escaped_invalid is False
    assert broken.false_rejection is False
    assert broken.budget_exhausted is True


def test_ground_truth_match_ignores_key_order_and_rejects_bool_coercion():
    assert output_matches_ground_truth('{"b": 1, "a": "x"}', {"a": "x", "b": 1})
    assert output_matches_ground_truth('  {"a": 1}\n', {"a": 1})
    assert output_matches_ground_truth('{"amount": 20.0}', {"amount": 20})
    assert output_matches_ground_truth('{"delayed": true}', {"delayed": True})
    assert output_matches_ground_truth(
        '{"items": ["Publicar nota", "Arquivar rascunho"]}',
        {"items": ["Publicar nota", "Arquivar rascunho"]},
    )
    assert not output_matches_ground_truth('{"delayed": 1}', {"delayed": True})
    assert not output_matches_ground_truth('{"delayed": 0}', {"delayed": False})
    assert not output_matches_ground_truth('{"a": "1"}', {"a": 1})
    assert not output_matches_ground_truth("not json", {"a": 1})
    assert not output_matches_ground_truth("[1]", {"a": 1})
    assert not output_matches_ground_truth('{"items": ["B", "A"]}', {"items": ["A", "B"]})

    task = next(item for item in load_eval_tasks() if item.id == "invoice_total_01")
    assert output_matches_ground_truth(task.examples.valid, task.ground_truth)
    assert not output_matches_ground_truth(task.examples.subtle, task.ground_truth)
    assert not output_matches_ground_truth(task.examples.malformed, task.ground_truth)


def test_nearest_rank_uses_one_based_ceil_position():
    assert nearest_rank([1, 2, 3, 4], 50) == 2
    assert nearest_rank([4, 1, 3, 2], 90) == 4
    assert nearest_rank([1, 2], 50) == 1
    assert nearest_rank([1, 2], 90) == 2
    assert nearest_rank([7], 50) == 7
    assert nearest_rank([7], 90) == 7
    with pytest.raises(ValueError, match="at least one"):
        nearest_rank([], 50)
    with pytest.raises(ValueError, match="percentile"):
        nearest_rank([1], 0)


def test_rates_use_only_scored_rows_and_do_not_sum_to_one():
    rows = [
        _ok(
            "json_schema",
            "structural_breakage",
            valid=True,
            matches=False,
            attempts=1,
            max_attempts=1,
            latency_ms=10,
        ),
        _ok(
            "json_schema",
            "structural_breakage",
            valid=False,
            matches=True,
            attempts=1,
            max_attempts=1,
            latency_ms=30,
        ),
        _ok(
            "json_schema",
            "unfaithful_extraction",
            valid=True,
            matches=True,
            attempts=2,
            max_attempts=3,
            latency_ms=20,
        ),
        _ok(
            "json_schema",
            "unfaithful_extraction",
            valid=False,
            matches=False,
            attempts=2,
            max_attempts=3,
            latency_ms=40,
        ),
        Observation(
            strategy="json_schema",
            category="structural_breakage",
            status="error",
            valid=False,
            matches_ground_truth=True,
            attempts=1,
            max_attempts=1,
            latency_ms=5,
        ),
    ]
    block = aggregate(rows)
    assert block.scored_count == 4
    assert block.error_count == 1
    assert block.acceptance_rate == pytest.approx(0.5)
    assert block.escaped_invalid_rate == pytest.approx(0.25)
    assert block.false_rejection_rate == pytest.approx(0.25)
    assert block.budget_exhaustion_rate == pytest.approx(0.25)
    assert block.attempts is not None
    assert block.attempts.mean == pytest.approx(1.5)
    assert block.attempts.p50 == 1
    assert block.attempts.p90 == 2
    assert block.latency_ms is not None
    assert block.latency_ms.mean == pytest.approx(25)
    assert block.latency_ms.p50 == 20
    assert block.latency_ms.p90 == 40


def test_error_rows_are_excluded_even_when_they_look_like_a_wrong_answer():
    block = aggregate(
        [
            Observation(
                strategy="pydantic",
                category="type_coercion_bait",
                status="error",
                valid=False,
                matches_ground_truth=True,
                attempts=3,
                max_attempts=3,
                latency_ms=9,
                prompt_tokens=100,
                completion_tokens=100,
                cost_usd=5,
            )
        ]
    )
    assert block.scored_count == 0
    assert block.error_count == 1
    assert block.acceptance_rate is None
    assert block.escaped_invalid_rate is None
    assert block.false_rejection_rate is None
    assert block.budget_exhaustion_rate is None
    assert block.attempts is None
    assert block.latency_ms is None
    assert block.prompt_tokens is None
    assert block.completion_tokens is None
    assert block.cost_usd is None


def test_empty_input_has_null_rates():
    block = aggregate([])
    assert block.scored_count == 0
    assert block.error_count == 0
    assert block.acceptance_rate is None
    assert block.prompt_tokens is None


def test_scored_row_missing_a_field_raises():
    with pytest.raises(ValueError, match="latency_ms"):
        aggregate(
            [
                Observation(
                    strategy="rules",
                    category="calendar_business_rule",
                    status="ok",
                    valid=True,
                    matches_ground_truth=True,
                    attempts=1,
                    max_attempts=1,
                )
            ]
        )


def test_tokens_and_cost_stay_null_unless_every_scored_row_has_them():
    complete = [
        _ok(
            "json_schema",
            "structural_breakage",
            valid=True,
            matches=True,
            attempts=1,
            max_attempts=1,
            latency_ms=1,
            prompt_tokens=10,
            completion_tokens=1,
            cost_usd=0.25,
        ),
        _ok(
            "json_schema",
            "structural_breakage",
            valid=True,
            matches=True,
            attempts=1,
            max_attempts=1,
            latency_ms=1,
            prompt_tokens=20,
            completion_tokens=2,
            cost_usd=0.75,
        ),
    ]
    both = aggregate(complete)
    assert both.prompt_tokens == 30
    assert both.completion_tokens == 3
    assert both.cost_usd == pytest.approx(1)

    missing_cost = aggregate(
        [
            complete[0],
            complete[1].model_copy(update={"cost_usd": None}),
        ]
    )
    assert missing_cost.prompt_tokens == 30
    assert missing_cost.completion_tokens == 3
    assert missing_cost.cost_usd is None

    missing_prompt = aggregate(
        [
            complete[0],
            complete[1].model_copy(update={"prompt_tokens": None}),
        ]
    )
    assert missing_prompt.prompt_tokens is None
    assert missing_prompt.completion_tokens == 3
    assert missing_prompt.cost_usd == pytest.approx(1)


def test_breakdown_does_not_mix_strategies_or_drop_an_error():
    rows = [
        _ok(
            "json_schema",
            "structural_breakage",
            valid=True,
            matches=False,
            attempts=1,
            max_attempts=1,
            latency_ms=10,
        ),
        _ok(
            "json_schema",
            "unfaithful_extraction",
            valid=True,
            matches=True,
            attempts=2,
            max_attempts=3,
            latency_ms=30,
        ),
        _ok(
            "pydantic",
            "structural_breakage",
            valid=False,
            matches=True,
            attempts=3,
            max_attempts=3,
            latency_ms=40,
        ),
        Observation(strategy="pydantic", category="structural_breakage", status="error"),
    ]
    summary = build_summary(
        rows,
        dataset="hand-built",
        backend="fake",
        scenario=None,
        strategies=["json_schema", "pydantic"],
        max_attempts=3,
        note="hand-built",
    )
    assert summary.record_count == 4
    assert summary.overall.scored_count == 3
    assert summary.overall.error_count == 1
    assert summary.overall.acceptance_rate == pytest.approx(2 / 3)
    assert summary.overall.escaped_invalid_rate == pytest.approx(1 / 3)
    assert summary.overall.false_rejection_rate == pytest.approx(1 / 3)

    schema = summary.by_strategy["json_schema"].overall
    assert schema.scored_count == 2
    assert schema.error_count == 0
    assert schema.acceptance_rate == 1
    assert schema.escaped_invalid_rate == pytest.approx(0.5)
    assert schema.attempts is not None
    assert schema.attempts.p50 == 1
    assert schema.attempts.p90 == 2

    pydantic = summary.by_strategy["pydantic"].overall
    assert pydantic.scored_count == 1
    assert pydantic.error_count == 1
    assert pydantic.acceptance_rate == 0
    assert pydantic.false_rejection_rate == 1
    assert pydantic.budget_exhaustion_rate == 1

    structural = summary.by_category["structural_breakage"]
    assert structural.scored_count == 2
    assert structural.error_count == 1
    assert structural.acceptance_rate == pytest.approx(0.5)
    faithful = summary.by_category["unfaithful_extraction"]
    assert faithful.scored_count == 1
    assert faithful.acceptance_rate == 1
    assert list(summary.by_category) == ["structural_breakage", "unfaithful_extraction"]
