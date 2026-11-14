"""Contrato do dataset v1: loader estrito, sondas por categoria e gabarito visível na fonte."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pytest

from guardrails_study.dataset import EVAL_CATEGORIES, EvalTask, eval_dataset_path, load_eval_tasks
from guardrails_study.guards import JsonSchemaGuard, PydanticGuard, RuleGuard
from guardrails_study.models import model_keys
from guardrails_study.tasks import script_from_labels

_TASKS = load_eval_tasks()
_ROOT = Path(__file__).resolve().parents[1]
_EMAIL = re.compile(r"[A-Za-z0-9._+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_COPIED_STRING_FIELDS = frozenset({"invoice_id", "email", "phone", "sku", "tracking", "order_id"})
_BUSINESS_OPS = frozenset({"calendar_date", "product_equals", "max_when_eq"})
_COERCION_TYPES = frozenset({"integer", "number", "boolean"})

EXPECTED_INVENTORY = (
    ("invoice_total_01", "structural_breakage", "invoice_total"),
    ("invoice_total_02", "type_coercion_bait", "invoice_total"),
    ("invoice_total_03", "type_coercion_bait", "invoice_total"),
    ("invoice_total_04", "type_coercion_bait", "invoice_total"),
    ("invoice_total_05", "unfaithful_extraction", "invoice_total"),
    ("due_date_01", "calendar_business_rule", "due_date"),
    ("due_date_02", "calendar_business_rule", "due_date"),
    ("due_date_03", "calendar_business_rule", "due_date"),
    ("due_date_04", "calendar_business_rule", "due_date"),
    ("due_date_05", "unfaithful_extraction", "due_date"),
    ("contact_email_01", "structural_breakage", "contact_email"),
    ("contact_email_02", "structural_breakage", "contact_email"),
    ("contact_email_03", "unfaithful_extraction", "contact_email"),
    ("contact_email_04", "unfaithful_extraction", "contact_email"),
    ("ticket_priority_01", "calendar_business_rule", "ticket_priority"),
    ("ticket_priority_02", "calendar_business_rule", "ticket_priority"),
    ("ticket_priority_03", "calendar_business_rule", "ticket_priority"),
    ("ticket_priority_04", "type_coercion_bait", "ticket_priority"),
    ("ticket_priority_05", "structural_breakage", "ticket_priority"),
    ("line_item_01", "calendar_business_rule", "line_item"),
    ("line_item_02", "calendar_business_rule", "line_item"),
    ("line_item_03", "calendar_business_rule", "line_item"),
    ("line_item_04", "type_coercion_bait", "line_item"),
    ("line_item_05", "unfaithful_extraction", "line_item"),
    ("phone_e164_01", "structural_breakage", "phone_e164"),
    ("phone_e164_02", "structural_breakage", "phone_e164"),
    ("phone_e164_03", "unfaithful_extraction", "phone_e164"),
    ("phone_e164_04", "unfaithful_extraction", "phone_e164"),
    ("shipment_delayed_01", "type_coercion_bait", "shipment_delayed"),
    ("shipment_delayed_02", "type_coercion_bait", "shipment_delayed"),
    ("shipment_delayed_03", "type_coercion_bait", "shipment_delayed"),
    ("shipment_delayed_04", "structural_breakage", "shipment_delayed"),
    ("shipment_delayed_05", "unfaithful_extraction", "shipment_delayed"),
    ("currency_amount_01", "type_coercion_bait", "currency_amount"),
    ("currency_amount_02", "type_coercion_bait", "currency_amount"),
    ("currency_amount_03", "structural_breakage", "currency_amount"),
    ("currency_amount_04", "unfaithful_extraction", "currency_amount"),
    ("action_items_01", "structural_breakage", "action_items"),
    ("action_items_02", "structural_breakage", "action_items"),
    ("action_items_03", "unfaithful_extraction", "action_items"),
    ("action_items_04", "unfaithful_extraction", "action_items"),
    ("order_status_01", "structural_breakage", "order_status"),
    ("order_status_02", "structural_breakage", "order_status"),
    ("order_status_03", "unfaithful_extraction", "order_status"),
    ("order_status_04", "unfaithful_extraction", "order_status"),
)

EXPECTED_CATEGORY_COUNTS = {
    "structural_breakage": 12,
    "type_coercion_bait": 10,
    "calendar_business_rule": 10,
    "unfaithful_extraction": 13,
}
EXPECTED_MODEL_COUNTS = {
    "invoice_total": 5,
    "due_date": 5,
    "contact_email": 4,
    "ticket_priority": 5,
    "line_item": 5,
    "phone_e164": 4,
    "shipment_delayed": 5,
    "currency_amount": 4,
    "action_items": 4,
    "order_status": 4,
}


def _tasks_in(category: str) -> list[EvalTask]:
    return [task for task in _TASKS if task.category == category]


def _record(task: EvalTask) -> dict:
    return task.model_dump(mode="json")


def _write(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "tasks.jsonl"
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_v1_inventory_is_stable():
    got = [(task.id, task.category, task.model_key) for task in _TASKS]
    assert got == list(EXPECTED_INVENTORY)
    ids = [task_id for task_id, _, _ in EXPECTED_INVENTORY]
    assert len(ids) == len(set(ids)) == 45


def test_counts_match_the_readme():
    categories = Counter(category for _, category, _ in EXPECTED_INVENTORY)
    models = Counter(model for _, _, model in EXPECTED_INVENTORY)
    assert dict(categories) == EXPECTED_CATEGORY_COUNTS
    assert dict(models) == EXPECTED_MODEL_COUNTS
    assert set(EXPECTED_CATEGORY_COUNTS) == set(EVAL_CATEGORIES)
    assert set(EXPECTED_MODEL_COUNTS) == model_keys()
    assert sum(EXPECTED_CATEGORY_COUNTS.values()) == 45
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    for name, count in {**EXPECTED_CATEGORY_COUNTS, **EXPECTED_MODEL_COUNTS}.items():
        assert f"| `{name}` | {count} |" in readme
    assert "não reporta resultado de modelo" in readme
    assert "não houve segundo anotador" in readme


def test_default_path_is_the_versioned_jsonl():
    path = eval_dataset_path()
    assert path.parts[-3:] == ("data", "eval", "tasks.v1.jsonl")
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert len(text.splitlines()) == 45
    emails = _EMAIL.findall(text)
    assert emails
    assert all(email.endswith("@example.com") for email in emails)
    for token in ("sk-", "sk-ant-", "api_key="):
        assert token not in text


@pytest.mark.parametrize("task", _TASKS, ids=lambda task: task.id)
def test_ground_truth_passes_deterministic_guards(task: EvalTask):
    raw = json.dumps(task.ground_truth)
    assert JsonSchemaGuard().check(raw, task).accepted
    assert PydanticGuard().check(raw, task).accepted
    assert RuleGuard().check(raw, task).accepted
    _assert_visible(task)


@pytest.mark.parametrize("task", _TASKS, ids=lambda task: task.id)
def test_malformed_examples_fail_json_and_deterministic_guards(task: EvalTask):
    with pytest.raises(json.JSONDecodeError):
        json.loads(task.examples.malformed)
    assert not JsonSchemaGuard().check(task.examples.malformed, task).accepted
    assert not PydanticGuard().check(task.examples.malformed, task).accepted
    assert not RuleGuard().check(task.examples.malformed, task).accepted


@pytest.mark.parametrize("task", _TASKS, ids=lambda task: task.id)
def test_subtle_examples_differ_from_truth_and_pass_deterministic_guards(task: EvalTask):
    parsed = json.loads(task.examples.subtle)
    assert isinstance(parsed, dict)
    assert parsed != task.ground_truth
    assert JsonSchemaGuard().check(task.examples.subtle, task).accepted
    assert PydanticGuard().check(task.examples.subtle, task).accepted
    assert RuleGuard().check(task.examples.subtle, task).accepted


@pytest.mark.parametrize("task", _TASKS, ids=lambda task: task.id)
def test_valid_example_matches_script_from_labels(task: EvalTask):
    valid, malformed, subtle = script_from_labels(task, ["valid", "malformed", "subtle"])
    assert valid == task.examples.valid
    assert json.loads(valid) == task.ground_truth
    assert malformed == task.examples.malformed
    assert subtle == task.examples.subtle


@pytest.mark.parametrize(
    "task",
    _tasks_in("structural_breakage"),
    ids=lambda task: task.id,
)
def test_structural_extra_field_fails_schema_and_pydantic_only(task: EvalTask):
    mutated = dict(task.ground_truth)
    mutated["extra_field"] = "nope"
    raw = json.dumps(mutated)
    assert not JsonSchemaGuard().check(raw, task).accepted
    assert not PydanticGuard().check(raw, task).accepted
    assert RuleGuard().check(raw, task).accepted


@pytest.mark.parametrize("task", _tasks_in("type_coercion_bait"), ids=lambda task: task.id)
def test_quoting_the_bait_scalar_fails_deterministic_guards(task: EvalTask):
    name, json_type = next(
        (key, spec["type"])
        for key, spec in task.json_schema["properties"].items()
        if spec.get("type") in _COERCION_TYPES
    )
    mutated = dict(task.ground_truth)
    value = mutated[name]
    if json_type == "boolean":
        mutated[name] = "true" if value else "false"
    else:
        mutated[name] = str(value)
    raw = json.dumps(mutated)
    assert not JsonSchemaGuard().check(raw, task).accepted
    assert not PydanticGuard().check(raw, task).accepted
    assert not RuleGuard().check(raw, task).accepted


@pytest.mark.parametrize(
    "task",
    _tasks_in("calendar_business_rule"),
    ids=lambda task: task.id,
)
def test_business_rule_probe_splits_schema_from_rules(task: EvalTask):
    rule = next(item for item in task.rules if item.op in _BUSINESS_OPS)
    mutated = dict(task.ground_truth)
    if rule.op == "calendar_date":
        mutated[rule.field or ""] = "2024-02-31"
    elif rule.op == "product_equals":
        mutated[rule.product or ""] = mutated[rule.product or ""] + 1
    else:
        mutated[rule.when_field or ""] = rule.when_value
        mutated[rule.field or ""] = int(rule.value or 0) + 1
    raw = json.dumps(mutated)
    assert JsonSchemaGuard().check(raw, task).accepted
    model_accepted = PydanticGuard().check(raw, task).accepted
    if rule.op == "calendar_date":
        assert not model_accepted
    else:
        assert model_accepted
    assert not RuleGuard().check(raw, task).accepted


@pytest.mark.parametrize(
    "task",
    _tasks_in("unfaithful_extraction"),
    ids=lambda task: task.id,
)
def test_unfaithful_subtle_swaps_a_string_that_is_in_the_source(task: EvalTask):
    subtle = json.loads(task.examples.subtle)
    assert _swaps_source_string(task.source_text, task.ground_truth, subtle)


def test_round_trip_of_one_record(tmp_path: Path):
    loaded = load_eval_tasks(_write(tmp_path, [_record(_TASKS[0])]))
    assert [task.id for task in loaded] == ["invoice_total_01"]


def test_loader_rejects_duplicate_id(tmp_path: Path):
    record = _record(_TASKS[0])
    with pytest.raises(ValueError, match="duplicate task id"):
        load_eval_tasks(_write(tmp_path, [record, record]))


def test_loader_rejects_subtle_equal_to_ground_truth(tmp_path: Path):
    record = _record(_TASKS[0])
    record["examples"]["subtle"] = record["examples"]["valid"]
    with pytest.raises(ValueError, match="subtle matches ground truth"):
        load_eval_tasks(_write(tmp_path, [record]))


def test_loader_rejects_malformed_that_is_json(tmp_path: Path):
    record = _record(_TASKS[0])
    record["examples"]["malformed"] = record["examples"]["valid"]
    with pytest.raises(ValueError, match="malformed is valid JSON"):
        load_eval_tasks(_write(tmp_path, [record]))


def test_loader_rejects_non_canonical_valid_example(tmp_path: Path):
    record = _record(_TASKS[0])
    record["examples"]["valid"] = record["examples"]["valid"] + " "
    with pytest.raises(ValueError, match="not the canonical ground truth"):
        load_eval_tasks(_write(tmp_path, [record]))


def test_loader_rejects_ground_truth_outside_the_schema(tmp_path: Path):
    record = _record(_TASKS[0])
    record["ground_truth"]["total_cents"] = -1
    record["examples"]["valid"] = json.dumps(
        record["ground_truth"], ensure_ascii=False, sort_keys=True
    )
    with pytest.raises(ValueError, match="ground truth rejected"):
        load_eval_tasks(_write(tmp_path, [record]))


def test_loader_rejects_unknown_category(tmp_path: Path):
    record = _record(_TASKS[0])
    record["category"] = "made_up"
    with pytest.raises(ValueError, match="unknown category"):
        load_eval_tasks(_write(tmp_path, [record]))


def test_loader_rejects_a_blank_line(tmp_path: Path):
    path = tmp_path / "tasks.jsonl"
    line = json.dumps(_record(_TASKS[0]), ensure_ascii=False)
    path.write_text(line + "\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="blank"):
        load_eval_tasks(path)


def test_loader_rejects_invalid_json_and_non_objects(tmp_path: Path):
    broken = tmp_path / "broken.jsonl"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1 is not JSON"):
        load_eval_tasks(broken)
    array = tmp_path / "array.jsonl"
    array.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a JSON object"):
        load_eval_tasks(array)


def test_loader_rejects_unknown_model_key(tmp_path: Path):
    record = _record(_TASKS[0])
    record["id"] = "missing_01"
    record["model_key"] = "missing"
    with pytest.raises(ValueError, match="unknown model_key"):
        load_eval_tasks(_write(tmp_path, [record]))


def test_loader_rejects_id_that_does_not_match_model_key(tmp_path: Path):
    record = _record(_TASKS[0])
    record["id"] = "due_date_01"
    with pytest.raises(ValueError, match="does not match model_key"):
        load_eval_tasks(_write(tmp_path, [record]))


def test_loader_rejects_empty_and_missing_files(tmp_path: Path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_eval_tasks(empty)
    with pytest.raises(ValueError, match="not found"):
        load_eval_tasks(tmp_path / "missing.jsonl")


def test_loader_rejects_category_without_its_probe(tmp_path: Path):
    email = _record(next(task for task in _TASKS if task.model_key == "contact_email"))
    email["category"] = "type_coercion_bait"
    with pytest.raises(ValueError, match="no scalar to quote"):
        load_eval_tasks(_write(tmp_path, [email]))
    email["category"] = "calendar_business_rule"
    with pytest.raises(ValueError, match="no business rule"):
        load_eval_tasks(_write(tmp_path, [email]))
    numeric = _record(next(task for task in _TASKS if task.id == "invoice_total_01"))
    numeric["category"] = "unfaithful_extraction"
    with pytest.raises(ValueError, match="does not swap a source string"):
        load_eval_tasks(_write(tmp_path, [numeric]))
    phone = _record(next(task for task in _TASKS if task.id == "phone_e164_01"))
    phone["json_schema"]["properties"]["phone"].pop("pattern")
    with pytest.raises(ValueError, match="no structural constraint"):
        load_eval_tasks(_write(tmp_path, [phone]))


def test_loader_rejects_subtle_that_breaks_a_deterministic_guard(tmp_path: Path):
    line = _record(next(task for task in _TASKS if task.id == "line_item_01"))
    subtle = json.loads(line["examples"]["subtle"])
    subtle["total_cents"] = subtle["total_cents"] + 1
    line["examples"]["subtle"] = json.dumps(subtle, ensure_ascii=False, sort_keys=True)
    with pytest.raises(ValueError, match="subtle rejected by rules"):
        load_eval_tasks(_write(tmp_path, [line]))
    due = _record(next(task for task in _TASKS if task.id == "due_date_01"))
    due["examples"]["subtle"] = json.dumps({"due_date": "2024-02-31"})
    with pytest.raises(ValueError, match="subtle rejected by pydantic"):
        load_eval_tasks(_write(tmp_path, [due]))


def test_loader_rejects_an_open_schema_and_unknown_fields(tmp_path: Path):
    record = _record(_TASKS[0])
    record["note"] = "nope"
    with pytest.raises(ValueError, match="does not match the eval task schema"):
        load_eval_tasks(_write(tmp_path, [record]))
    opened = _record(_TASKS[0])
    opened["json_schema"]["additionalProperties"] = True
    with pytest.raises(ValueError, match="closed object"):
        load_eval_tasks(_write(tmp_path, [opened]))


def _assert_visible(task: EvalTask) -> None:
    for key, value in task.ground_truth.items():
        if key in _COPIED_STRING_FIELDS:
            assert value in task.source_text
        elif key == "due_date":
            assert isinstance(value, str)
            if task.category == "calendar_business_rule":
                year, month, day = value.split("-")
                assert f"{day}/{month}/{year}" in task.source_text
            else:
                assert value in task.source_text
        elif key == "items":
            assert all(item in task.source_text for item in value)
        elif isinstance(value, bool):
            phrase = "há atraso" if value else "sem atraso"
            assert phrase in task.source_text
        elif isinstance(value, int | float):
            assert str(value) in task.source_text


def _distinct_in_source(source: str, left: str, right: str) -> bool:
    if left in right or right in left:
        return False
    return left in source and right in source


def _swaps_source_string(source: str, truth: dict, subtle: dict) -> bool:
    for key, original in truth.items():
        other = subtle.get(key)
        if other == original:
            continue
        if (
            isinstance(original, str)
            and isinstance(other, str)
            and _distinct_in_source(source, original, other)
        ):
            return True
        if isinstance(original, list) and isinstance(other, list):
            removed = [item for item in original if item not in other]
            added = [item for item in other if item not in original]
            pairs = [(left, right) for left in removed for right in added]
            if pairs and all(
                isinstance(left, str)
                and isinstance(right, str)
                and _distinct_in_source(source, left, right)
                for left, right in pairs
            ):
                return True
    return False
