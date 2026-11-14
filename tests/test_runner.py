"""Orçamento, re-ask e registro. valid é aceite da guarda, não acordo com o gabarito."""

from __future__ import annotations

import json

import pytest

from guardrails_study.config import Settings
from guardrails_study.guards import JsonSchemaGuard
from guardrails_study.llm import ScriptedLLM, ScriptedOutput, TokenUsage
from guardrails_study.runner import attempts_for, run_strategy, run_task
from guardrails_study.tasks import script_from_labels
from tests.helpers import get_task


def test_retry_budget_stops_and_records_the_rejecting_guard():
    task = get_task("invoice_total")
    generator = ScriptedLLM(script_from_labels(task, ["malformed", "malformed", "malformed"]))
    result = run_task(
        task,
        generator,
        JsonSchemaGuard(),
        strategy="json_schema",
        max_attempts=3,
    )
    assert result.valid is False
    assert result.attempts == 3
    assert result.rejected_by == "json_schema"
    assert len(generator.calls) == 3
    assert result.final_output == task.examples.malformed
    assert result.latency_ms == pytest.approx(sum(item.latency_ms for item in result.history))
    assert result.latency_ms >= 0
    assert result.prompt_tokens is None
    assert result.cost_usd is None


def test_success_on_second_attempt_does_not_spend_the_rest_of_the_budget():
    task = get_task("invoice_total")
    generator = ScriptedLLM(script_from_labels(task, ["malformed", "valid", "subtle"]))
    result = run_task(
        task,
        generator,
        JsonSchemaGuard(),
        strategy="json_schema",
        max_attempts=3,
    )
    assert result.valid is True
    assert result.attempts == 2
    assert result.rejected_by is None
    assert len(generator.calls) == 2
    assert json.loads(result.final_output) == task.ground_truth
    repair = generator.calls[1][-1]["content"]
    assert "json_schema" in repair
    assert "malformed_json" in repair
    assert "ground_truth" not in repair
    dumped = json.dumps(task.ground_truth, ensure_ascii=False, sort_keys=True)
    assert dumped not in repair


def test_max_attempts_below_one_raises():
    task = get_task("invoice_total")
    with pytest.raises(ValueError, match="max_attempts"):
        run_task(
            task,
            ScriptedLLM([]),
            JsonSchemaGuard(),
            strategy="json_schema",
            max_attempts=0,
        )


def test_empty_script_raises_instead_of_inventing_output():
    task = get_task("invoice_total")
    with pytest.raises(RuntimeError, match="queue is empty"):
        run_task(
            task,
            ScriptedLLM([]),
            JsonSchemaGuard(),
            strategy="json_schema",
            max_attempts=1,
        )


def test_json_schema_preset_is_single_shot():
    task = get_task("order_status")
    settings = Settings(max_attempts=3)
    assert attempts_for("json_schema", settings) == 1
    generator = ScriptedLLM(script_from_labels(task, ["malformed", "valid"]))
    result = run_strategy("json_schema", task, generator, settings)
    assert result.attempts == 1
    assert result.valid is False
    assert len(generator.calls) == 1


def test_pydantic_preset_reasks_until_valid():
    task = get_task("invoice_total")
    settings = Settings(max_attempts=3)
    generator = ScriptedLLM(script_from_labels(task, ["malformed", "valid"]))
    result = run_strategy("pydantic", task, generator, settings)
    assert result.strategy == "pydantic"
    assert result.valid is True
    assert result.attempts == 2
    assert result.task_id == task.id


def test_rules_preset_rejects_product_mismatch_and_retries():
    task = get_task("line_item")
    bad = json.dumps({"sku": "SKU-100", "qty": 2, "unit_cents": 500, "total_cents": 999})
    good = json.dumps(task.ground_truth)
    generator = ScriptedLLM([bad, good])
    result = run_strategy("rules", task, generator, Settings(max_attempts=2))
    assert result.attempts == 2
    assert result.valid is True
    assert "product_equals" in generator.calls[1][-1]["content"]


def test_second_llm_false_rejection_and_slip_through_are_both_recordable():
    task = get_task("order_status")
    truth = json.dumps(task.ground_truth, sort_keys=True)
    settings = Settings(max_attempts=1)
    rejector = ScriptedLLM(['{"accept": false, "reason": "discordo do texto"}'])
    false_reject = run_strategy(
        "second_llm",
        task,
        ScriptedLLM([truth]),
        settings,
        checker=rejector,
    )
    assert false_reject.valid is False
    assert false_reject.rejected_by == "second_llm"
    assert json.loads(false_reject.final_output) == task.ground_truth

    acceptor = ScriptedLLM(['{"accept": true, "reason": "ok"}'])
    slipped = run_strategy(
        "second_llm",
        task,
        ScriptedLLM([task.examples.subtle]),
        settings,
        checker=acceptor,
    )
    assert slipped.valid is True
    assert json.loads(slipped.final_output) != task.ground_truth


def test_tokens_and_cost_require_complete_usage_and_prices():
    task = get_task("invoice_total")
    raw = json.dumps(task.ground_truth)
    generator = ScriptedLLM(
        [ScriptedOutput(text=raw, usage=TokenUsage(prompt_tokens=17, completion_tokens=8))]
    )
    priced = Settings(price_prompt_per_million=1_000_000, price_completion_per_million=1_000_000)
    result = run_task(
        task,
        generator,
        JsonSchemaGuard(),
        strategy="json_schema",
        max_attempts=1,
        settings=priced,
    )
    assert result.prompt_tokens == 17
    assert result.completion_tokens == 8
    assert result.cost_usd == pytest.approx(25)

    unpriced = run_task(
        task,
        ScriptedLLM(
            [ScriptedOutput(text=raw, usage=TokenUsage(prompt_tokens=17, completion_tokens=8))]
        ),
        JsonSchemaGuard(),
        strategy="json_schema",
        max_attempts=1,
        settings=Settings(),
    )
    assert unpriced.prompt_tokens == 17
    assert unpriced.cost_usd is None

    missing_usage = run_task(
        task,
        ScriptedLLM([raw]),
        JsonSchemaGuard(),
        strategy="json_schema",
        max_attempts=1,
        settings=priced,
    )
    assert missing_usage.prompt_tokens is None
    assert missing_usage.cost_usd is None


def test_failed_attempt_tokens_count_and_checker_tokens_add_up():
    task = get_task("order_status")
    bad = task.examples.malformed
    good = json.dumps(task.ground_truth)
    generator = ScriptedLLM(
        [
            ScriptedOutput(text=bad, usage=TokenUsage(prompt_tokens=10, completion_tokens=1)),
            ScriptedOutput(text=good, usage=TokenUsage(prompt_tokens=12, completion_tokens=4)),
        ]
    )
    checker = ScriptedLLM(
        [
            ScriptedOutput(
                text='{"accept": false, "reason": "quebrado"}',
                usage=TokenUsage(prompt_tokens=3, completion_tokens=2),
            ),
            ScriptedOutput(
                text='{"accept": true, "reason": "ok"}',
                usage=TokenUsage(prompt_tokens=4, completion_tokens=2),
            ),
        ]
    )
    result = run_strategy(
        "second_llm",
        task,
        generator,
        Settings(max_attempts=2, price_prompt_per_million=0, price_completion_per_million=0),
        checker=checker,
    )
    assert result.valid is True
    assert result.attempts == 2
    assert result.prompt_tokens == 10 + 3 + 12 + 4
    assert result.completion_tokens == 1 + 2 + 4 + 2
    assert result.cost_usd == pytest.approx(0)


def test_checker_without_usage_nulls_the_whole_attempt():
    task = get_task("order_status")
    generator = ScriptedLLM(
        [
            ScriptedOutput(
                text=json.dumps(task.ground_truth),
                usage=TokenUsage(prompt_tokens=9, completion_tokens=1),
            )
        ]
    )
    checker = ScriptedLLM(['{"accept": true, "reason": "ok"}'])
    result = run_strategy(
        "second_llm",
        task,
        generator,
        Settings(max_attempts=1, price_prompt_per_million=2, price_completion_per_million=3),
        checker=checker,
    )
    assert result.valid is True
    assert result.prompt_tokens is None
    assert result.cost_usd is None


def test_result_record_exposes_the_study_fields():
    task = get_task("invoice_total")
    result = run_task(
        task,
        ScriptedLLM([json.dumps(task.ground_truth)]),
        JsonSchemaGuard(),
        strategy="json_schema",
        max_attempts=1,
    )
    dumped = result.model_dump()
    for key in (
        "final_output",
        "valid",
        "attempts",
        "rejected_by",
        "latency_ms",
        "prompt_tokens",
        "completion_tokens",
        "cost_usd",
    ):
        assert key in dumped


def test_unknown_strategy_preset_raises():
    with pytest.raises(ValueError, match="unknown strategy"):
        attempts_for("stack", Settings())
