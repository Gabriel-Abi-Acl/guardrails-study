"""Cada guarda rejeita o próprio modo de falha e não repara o texto."""

from __future__ import annotations

import json

import pytest

from guardrails_study.guards import (
    JsonSchemaGuard,
    PydanticGuard,
    RuleGuard,
    SecondLLMGuard,
    build_guard,
)
from guardrails_study.llm import ScriptedLLM, ScriptedOutput, TokenUsage
from tests.helpers import get_task


def test_schema_accepts_structurally_valid_wrong_invoice_total():
    task = get_task("invoice_total")
    verdict = JsonSchemaGuard().check(task.examples.subtle, task)
    assert verdict.accepted
    assert json.loads(task.examples.subtle) != task.ground_truth


def test_impossible_date_splits_schema_from_pydantic_and_rules():
    task = get_task("due_date")
    raw = json.dumps({"due_date": "2024-02-31"})
    assert JsonSchemaGuard().check(raw, task).accepted
    assert not PydanticGuard().check(raw, task).accepted
    rule = RuleGuard().check(raw, task)
    assert not rule.accepted
    assert rule.reason is not None
    assert rule.reason.startswith("rule_failed:")


def test_string_integer_is_not_coerced():
    task = get_task("invoice_total")
    raw = json.dumps({"invoice_id": "INV-1042", "total_cents": "1990"})
    assert not JsonSchemaGuard().check(raw, task).accepted
    model = PydanticGuard().check(raw, task)
    assert not model.accepted
    assert model.reason is not None
    assert model.reason.startswith("model_mismatch:")
    assert not RuleGuard().check(raw, task).accepted


def test_string_boolean_is_not_coerced():
    task = get_task("shipment_delayed")
    raw = json.dumps({"delayed": "false", "tracking": "TRK-9"})
    assert not JsonSchemaGuard().check(raw, task).accepted
    assert not PydanticGuard().check(raw, task).accepted
    assert not RuleGuard().check(raw, task).accepted


def test_product_mismatch_is_rule_only():
    task = get_task("line_item")
    raw = json.dumps({"sku": "SKU-100", "qty": 2, "unit_cents": 500, "total_cents": 999})
    assert JsonSchemaGuard().check(raw, task).accepted
    assert PydanticGuard().check(raw, task).accepted
    verdict = RuleGuard().check(raw, task)
    assert not verdict.accepted
    assert verdict.reason is not None
    assert "product_equals" in verdict.reason


def test_high_priority_sla_above_cap_is_rule_only():
    task = get_task("ticket_priority")
    raw = json.dumps({"priority": "high", "sla_hours": 48})
    assert JsonSchemaGuard().check(raw, task).accepted
    assert PydanticGuard().check(raw, task).accepted
    assert not RuleGuard().check(raw, task).accepted


def test_low_priority_does_not_apply_sla_cap():
    task = get_task("ticket_priority")
    raw = json.dumps({"priority": "low", "sla_hours": 100})
    assert RuleGuard().check(raw, task).accepted


def test_high_priority_sla_at_cap_passes_rules():
    task = get_task("ticket_priority")
    raw = json.dumps({"priority": "high", "sla_hours": 4})
    assert RuleGuard().check(raw, task).accepted


def test_extra_field_fails_schema_and_model_but_not_listed_rules():
    task = get_task("order_status")
    payload = dict(task.ground_truth)
    payload["note"] = "extra"
    raw = json.dumps(payload)
    assert not JsonSchemaGuard().check(raw, task).accepted
    assert not PydanticGuard().check(raw, task).accepted
    assert RuleGuard().check(raw, task).accepted


def test_empty_output_and_json_array_are_rejected():
    task = get_task("order_status")
    for raw in ("", "[]"):
        for guard in (JsonSchemaGuard(), PydanticGuard(), RuleGuard()):
            verdict = guard.check(raw, task)
            assert not verdict.accepted
            assert verdict.guard_name == guard.name


def test_unknown_model_key_is_a_programming_error():
    task = get_task("invoice_total").model_copy(update={"model_key": "missing"})
    with pytest.raises(KeyError, match="unknown model_key"):
        PydanticGuard().check(json.dumps(task.ground_truth), task)


def test_second_llm_accepts_only_a_boolean_contract():
    task = get_task("order_status")
    checker = ScriptedLLM(
        [
            '{"accept": true, "reason": "ok", "notes": "ignored"}',
            '{"accept": "true", "reason": "ok"}',
            "not json",
            '{"accept": false, "reason": "status ignores the source"}',
        ]
    )
    guard = SecondLLMGuard(checker)
    assert guard.check(task.examples.subtle, task).accepted
    malformed = guard.check(task.examples.subtle, task)
    assert not malformed.accepted
    assert malformed.reason == "checker_malformed"
    prose = guard.check(task.examples.subtle, task)
    assert prose.reason == "checker_malformed"
    rejected = guard.check(json.dumps(task.ground_truth), task)
    assert not rejected.accepted
    assert rejected.reason is not None
    assert rejected.reason.startswith("checker_rejected:")


def test_second_llm_records_checker_usage_and_missing_usage():
    task = get_task("order_status")
    with_usage = SecondLLMGuard(
        ScriptedLLM(
            [
                ScriptedOutput(
                    text='{"accept": true, "reason": "ok"}',
                    usage=TokenUsage(prompt_tokens=7, completion_tokens=3),
                )
            ]
        )
    )
    verdict = with_usage.check("{}", task)
    assert verdict.extra_usage is not None
    assert verdict.extra_usage.prompt_tokens == 7
    assert not verdict.extra_call_without_usage

    without = SecondLLMGuard(ScriptedLLM(['{"accept": true, "reason": "ok"}']))
    missing = without.check("{}", task)
    assert missing.extra_usage is None
    assert missing.extra_call_without_usage


def test_checker_prompt_does_not_include_ground_truth_payload():
    task = get_task("order_status")
    checker = ScriptedLLM(['{"accept": true, "reason": "ok"}'])
    SecondLLMGuard(checker).check('{"order_id": "ORD-77", "status": "pending"}', task)
    prompt = json.dumps(checker.calls[0], ensure_ascii=False)
    assert "ground_truth" not in prompt
    assert json.dumps(task.ground_truth, ensure_ascii=False, sort_keys=True) not in prompt


def test_build_guard_requires_checker_only_for_second_llm():
    assert build_guard("json_schema").name == "json_schema"
    assert build_guard("pydantic").name == "pydantic"
    assert build_guard("rules").name == "rules"
    with pytest.raises(ValueError, match="checker"):
        build_guard("second_llm")
    with pytest.raises(ValueError, match="unknown strategy"):
        build_guard("stack")
