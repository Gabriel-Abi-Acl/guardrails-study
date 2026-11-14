"""Carrega o dataset versionado e recusa linha que quebre o contrato.

O gabarito e o subtle passam nas três guardas determinísticas; o subtle mesmo assim
difere do gabarito. O malformed não é JSON. A categoria é sonda à parte, não o subtle.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, NamedTuple

from pydantic import BaseModel, ConfigDict, ValidationError

from guardrails_study.guards import JsonSchemaGuard, PydanticGuard, RuleGuard
from guardrails_study.tasks import Rule, Task, validate_task_definition

EVAL_CATEGORIES: tuple[str, ...] = (
    "structural_breakage",
    "type_coercion_bait",
    "calendar_business_rule",
    "unfaithful_extraction",
)

_BUSINESS_OPS = ("calendar_date", "product_equals", "max_when_eq")
_COERCION_TYPES = ("integer", "number", "boolean")
_STRUCTURAL_KEYS = ("pattern", "enum", "minItems", "maxItems", "minLength", "maxLength")
_ID = re.compile(r"^[a-z][a-z0-9_]*$")


class EvalExamples(BaseModel):
    model_config = ConfigDict(extra="forbid")
    valid: str
    malformed: str
    subtle: str


class EvalTask(Task):
    """Tarefa do dataset. category é o estresse primário, não um quarto exemplo."""

    category: str
    examples: EvalExamples


class _Guards(NamedTuple):
    json_schema: JsonSchemaGuard
    pydantic: PydanticGuard
    rules: RuleGuard


def eval_dataset_path() -> Path:
    """Caminho do v1 na raiz do repositório. O arquivo não vai no wheel."""
    return Path(__file__).resolve().parents[1] / "data" / "eval" / "tasks.v1.jsonl"


def load_eval_tasks(path: Path | None = None) -> list[EvalTask]:
    """Lê o JSONL e aplica o contrato. Uma linha inválida invalida o arquivo inteiro."""
    dataset = eval_dataset_path() if path is None else path
    if not dataset.is_file():
        raise ValueError(f"eval dataset not found: {dataset}")
    text = dataset.read_text(encoding="utf-8")
    if text.strip() == "":
        raise ValueError("eval dataset is empty")
    guards = _Guards(JsonSchemaGuard(), PydanticGuard(), RuleGuard())
    tasks: list[EvalTask] = []
    seen: set[str] = set()
    for index, line in enumerate(text.splitlines(), start=1):
        if line.strip() == "":
            raise ValueError(f"line {index} is blank")
        task = _parse_line(index, line)
        if task.id in seen:
            raise ValueError(f"duplicate task id: {task.id}")
        seen.add(task.id)
        _validate_task(task, guards)
        tasks.append(task)
    return tasks


def _parse_line(index: int, line: str) -> EvalTask:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"line {index} is not JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"line {index} is not a JSON object")
    try:
        return EvalTask.model_validate(payload)
    except ValidationError as exc:
        message = str(exc.errors()[0].get("msg", "invalid"))
        raise ValueError(f"line {index} does not match the eval task schema: {message}") from exc


def _validate_task(task: EvalTask, guards: _Guards) -> None:
    if _ID.fullmatch(task.id) is None:
        raise ValueError(f"invalid task id: {task.id}")
    if not task.id.startswith(f"{task.model_key}_"):
        raise ValueError(f"{task.id} id does not match model_key")
    validate_task_definition(task)
    properties = _assert_closed_object(task)
    _assert_examples(task, guards)
    _assert_category_probe(task, properties, guards)


def _assert_closed_object(task: EvalTask) -> dict[str, dict[str, Any]]:
    schema = task.json_schema
    if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        raise ValueError(f"{task.id} schema must be a closed object")
    properties = schema.get("properties")
    required = schema.get("required")
    if not isinstance(properties, dict) or not properties:
        raise ValueError(f"{task.id} schema has no properties")
    if not isinstance(required, list) or set(required) != set(properties):
        raise ValueError(f"{task.id} required keys diverged from properties")
    if set(task.ground_truth) != set(properties):
        raise ValueError(f"{task.id} ground truth keys diverged from the schema")
    specs: dict[str, dict[str, Any]] = {}
    for name, spec in properties.items():
        if not isinstance(spec, dict):
            raise ValueError(f"{task.id} property {name} is not an object")
        specs[name] = spec
    return specs


def _assert_examples(task: EvalTask, guards: _Guards) -> None:
    # valid é o mesmo texto que script_from_labels emite. Outra serialização é outro roteiro.
    canonical = json.dumps(task.ground_truth, ensure_ascii=False, sort_keys=True)
    if task.examples.valid != canonical:
        raise ValueError(f"{task.id} examples.valid is not the canonical ground truth")
    try:
        json.loads(task.examples.malformed)
    except json.JSONDecodeError:
        pass
    else:
        raise ValueError(f"{task.id} malformed is valid JSON")
    try:
        subtle = json.loads(task.examples.subtle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{task.id} subtle is not JSON") from exc
    if not isinstance(subtle, dict):
        raise ValueError(f"{task.id} subtle is not a JSON object")
    if subtle == task.ground_truth:
        raise ValueError(f"{task.id} subtle matches ground truth")
    _assert_accepted(task, "ground truth", task.examples.valid, guards)
    _assert_accepted(task, "subtle", task.examples.subtle, guards)


def _assert_accepted(task: EvalTask, label: str, raw: str, guards: _Guards) -> None:
    for guard in guards:
        verdict = guard.check(raw, task)
        if not verdict.accepted:
            raise ValueError(f"{task.id} {label} rejected by {guard.name}: {verdict.reason}")


def _assert_category_probe(
    task: EvalTask,
    properties: dict[str, dict[str, Any]],
    guards: _Guards,
) -> None:
    """A categoria tem de ser reproduzível. O subtle não serve de sonda: ele passa nas guardas."""
    if task.category == "structural_breakage":
        _assert_structural(task, properties, guards)
    elif task.category == "type_coercion_bait":
        _assert_coercion(task, properties, guards)
    elif task.category == "calendar_business_rule":
        _assert_business_rule(task, guards)
    elif task.category == "unfaithful_extraction":
        _assert_unfaithful(task)
    else:
        raise ValueError(f"{task.id} has unknown category {task.category}")


def _assert_structural(
    task: EvalTask,
    properties: dict[str, dict[str, Any]],
    guards: _Guards,
) -> None:
    if not any(key in spec for spec in properties.values() for key in _STRUCTURAL_KEYS):
        raise ValueError(f"{task.id} structural_breakage schema has no structural constraint")
    mutated = dict(task.ground_truth)
    mutated["extra_field"] = "nope"
    raw = json.dumps(mutated)
    if guards.json_schema.check(raw, task).accepted or guards.pydantic.check(raw, task).accepted:
        raise ValueError(f"{task.id} structural_breakage extra field was accepted")
    # Campo extra passa na regra de propósito: a regra não fecha o objeto.
    if not guards.rules.check(raw, task).accepted:
        raise ValueError(f"{task.id} structural_breakage extra field was rejected by rules")


def _assert_coercion(
    task: EvalTask,
    properties: dict[str, dict[str, Any]],
    guards: _Guards,
) -> None:
    found: tuple[str, str] | None = None
    for name, spec in properties.items():
        json_type = spec.get("type")
        if json_type in _COERCION_TYPES:
            found = (name, str(json_type))
            break
    if found is None:
        raise ValueError(f"{task.id} type_coercion_bait has no scalar to quote")
    name, json_type = found
    mutated = dict(task.ground_truth)
    mutated[name] = _quote_scalar(task.id, mutated[name], json_type)
    raw = json.dumps(mutated)
    for guard in guards:
        if guard.check(raw, task).accepted:
            raise ValueError(
                f"{task.id} type_coercion_bait quoted {name} was accepted by {guard.name}"
            )


def _quote_scalar(task_id: str, value: object, json_type: str) -> str:
    if json_type == "boolean":
        if not isinstance(value, bool):
            raise ValueError(f"{task_id} type_coercion_bait boolean field is not a boolean")
        return "true" if value else "false"
    if json_type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{task_id} type_coercion_bait integer field is not an integer")
        return str(value)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{task_id} type_coercion_bait number field is not a number")
    return str(value)


def _assert_business_rule(task: EvalTask, guards: _Guards) -> None:
    rule = next((item for item in task.rules if item.op in _BUSINESS_OPS), None)
    if rule is None:
        raise ValueError(f"{task.id} calendar_business_rule has no business rule")
    mutated = dict(task.ground_truth)
    _apply_business_probe(task.id, mutated, rule)
    raw = json.dumps(mutated)
    if not guards.json_schema.check(raw, task).accepted:
        raise ValueError(f"{task.id} calendar_business_rule probe was rejected by schema")
    model_accepted = guards.pydantic.check(raw, task).accepted
    if rule.op == "calendar_date":
        if model_accepted:
            raise ValueError(f"{task.id} impossible date passed pydantic")
    elif not model_accepted:
        raise ValueError(f"{task.id} calendar_business_rule probe was rejected by pydantic")
    if guards.rules.check(raw, task).accepted:
        raise ValueError(f"{task.id} calendar_business_rule probe was accepted by rules")


def _apply_business_probe(task_id: str, mutated: dict[str, Any], rule: Rule) -> None:
    if rule.op == "calendar_date":
        mutated[rule.field or ""] = "2024-02-31"
        return
    if rule.op == "product_equals":
        field = rule.product or ""
        current = mutated.get(field)
        if isinstance(current, bool) or not isinstance(current, int):
            raise ValueError(f"{task_id} product field is not an integer")
        mutated[field] = current + 1
        return
    if rule.value is None or rule.value != int(rule.value):
        raise ValueError(f"{task_id} max_when_eq cap is not an integer")
    mutated[rule.when_field or ""] = rule.when_value
    mutated[rule.field or ""] = int(rule.value) + 1


def _assert_unfaithful(task: EvalTask) -> None:
    subtle = json.loads(task.examples.subtle)
    if _swaps_source_string(task.source_text, task.ground_truth, subtle):
        return
    raise ValueError(f"{task.id} unfaithful_extraction subtle does not swap a source string")


def _swaps_source_string(source: str, truth: dict[str, Any], subtle: dict[str, Any]) -> bool:
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
            if not removed or not added:
                continue
            pairs = [(left, right) for left in removed for right in added]
            if all(
                isinstance(left, str)
                and isinstance(right, str)
                and _distinct_in_source(source, left, right)
                for left, right in pairs
            ):
                return True
    return False


def _distinct_in_source(source: str, left: str, right: str) -> bool:
    # Um não pode ser pedaço do outro: "TRK-30" dentro de "TRK-305" não é distrator.
    if left in right or right in left:
        return False
    return left in source and right in source
