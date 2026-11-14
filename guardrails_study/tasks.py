"""Carrega a fixture sintética e monta roteiros do fake a partir dos rótulos."""

from __future__ import annotations

import json
import re
from importlib.resources import files
from typing import Any, Literal

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import BaseModel, ConfigDict, Field

from guardrails_study.models import model_keys

ScriptLabel = Literal["valid", "malformed", "subtle"]

_RULE_FIELDS: dict[str, tuple[str, ...]] = {
    "regex": ("field", "pattern"),
    "gte": ("field", "value", "numeric"),
    "type": ("field", "json_type"),
    "calendar_date": ("field",),
    "enum": ("field", "choices"),
    "max_when_eq": ("field", "when_field", "when_value", "value"),
    "product_equals": ("factors", "product"),
    "text_length": ("field", "min_length", "max_length"),
    "list_len": ("field", "minimum", "maximum"),
}


class Rule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    op: str
    field: str | None = None
    pattern: str | None = None
    value: float | None = None
    numeric: Literal["integer", "number"] | None = None
    json_type: Literal["string", "integer", "number", "boolean"] | None = None
    choices: list[str] | None = None
    when_field: str | None = None
    when_value: str | None = None
    factors: list[str] | None = None
    product: str | None = None
    min_length: int | None = None
    max_length: int | None = None
    minimum: int | None = None
    maximum: int | None = None


class TaskExamples(BaseModel):
    model_config = ConfigDict(extra="forbid")
    malformed: str
    subtle: str


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    instruction: str
    source_text: str
    json_schema: dict[str, Any]
    ground_truth: dict[str, Any]
    model_key: str
    rules: list[Rule] = Field(min_length=1)
    examples: TaskExamples


def load_tasks() -> list[Task]:
    """Lê tasks.json do pacote. Schema inválido ou id repetido é erro da fixture."""
    raw = files("guardrails_study.fixtures").joinpath("tasks.json").read_text(encoding="utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, list):
        raise ValueError("tasks.json must be a list")
    tasks = [Task.model_validate(item) for item in payload]
    seen: set[str] = set()
    for task in tasks:
        if task.id in seen:
            raise ValueError(f"duplicate task id: {task.id}")
        seen.add(task.id)
        validate_task_definition(task)
    return tasks


def validate_task_definition(task: Task) -> None:
    """Confere model_key, schema e regras. Não olha exemplos nem gabarito contra o texto."""
    if task.model_key not in model_keys():
        raise ValueError(f"{task.id} has unknown model_key {task.model_key}")
    try:
        Draft202012Validator.check_schema(task.json_schema)
    except SchemaError as exc:
        raise ValueError(f"{task.id} has an invalid JSON Schema") from exc
    for rule in task.rules:
        _check_rule(task.id, rule)


def task_by_id(tasks: list[Task], task_id: str) -> Task:
    for task in tasks:
        if task.id == task_id:
            return task
    raise KeyError(f"unknown task id: {task_id}")


def script_from_labels(task: Task, labels: list[ScriptLabel]) -> list[str]:
    """Traduz rótulos em textos da fila. valid é o gabarito serializado, não uma amostra nova."""
    outputs: list[str] = []
    for label in labels:
        if label == "valid":
            outputs.append(json.dumps(task.ground_truth, ensure_ascii=False, sort_keys=True))
        elif label == "malformed":
            outputs.append(task.examples.malformed)
        elif label == "subtle":
            outputs.append(task.examples.subtle)
        else:
            raise ValueError(f"unknown script label: {label}")
    return outputs


def _check_rule(task_id: str, rule: Rule) -> None:
    required = _RULE_FIELDS.get(rule.op)
    if required is None:
        raise ValueError(f"{task_id} has unknown rule op {rule.op}")
    for name in required:
        if getattr(rule, name) is None:
            raise ValueError(f"{task_id} rule {rule.op} missing {name}")
    if rule.op == "regex":
        re.compile(rule.pattern or "")
    if rule.op == "enum" and not rule.choices:
        raise ValueError(f"{task_id} enum rule has no choices")
    if rule.op == "product_equals" and (rule.factors is None or len(rule.factors) < 2):
        raise ValueError(f"{task_id} product_equals needs two factors")
    if rule.op == "text_length" and (rule.min_length or 0) > (rule.max_length or 0):
        raise ValueError(f"{task_id} text_length range is inverted")
    if rule.op == "list_len" and (rule.minimum or 0) > (rule.maximum or 0):
        raise ValueError(f"{task_id} list_len range is inverted")
