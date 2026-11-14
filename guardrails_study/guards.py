"""Quatro guardas independentes. Nenhuma repara a saída nem vê o gabarito.

`accepted` aqui é decisão da guarda. Concordância com o gabarito fica para o PR3.
"""

from __future__ import annotations

import datetime
import json
import re
from typing import Any, Protocol

from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict, ValidationError

from guardrails_study.llm import LLMClient, TokenUsage
from guardrails_study.models import model_for
from guardrails_study.tasks import Rule, Task

_REASON_LIMIT = 500


class GuardVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    accepted: bool
    guard_name: str
    reason: str | None = None
    extra_usage: TokenUsage | None = None
    # True quando a guarda chamou um LLM e ele não informou uso. Não preenche zero.
    extra_call_without_usage: bool = False


class Guard(Protocol):
    name: str

    def check(self, raw: str, task: Task) -> GuardVerdict:
        """Aceita ou rejeita o texto bruto. Não levanta erro de validação da saída."""


def parse_json_object(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"malformed_json: {exc.msg}"
    if not isinstance(value, dict):
        return None, "json_not_object"
    return value, None


def _clip(text: str) -> str:
    if len(text) <= _REASON_LIMIT:
        return text
    return text[: _REASON_LIMIT - 3] + "..."


def _reject(guard_name: str, reason: str, **kwargs: Any) -> GuardVerdict:
    return GuardVerdict(accepted=False, guard_name=guard_name, reason=_clip(reason), **kwargs)


def _accept(guard_name: str, **kwargs: Any) -> GuardVerdict:
    return GuardVerdict(accepted=True, guard_name=guard_name, reason=None, **kwargs)


class JsonSchemaGuard:
    """Valida o objeto contra o JSON Schema da tarefa. format do draft não é assertado."""

    name = "json_schema"

    def check(self, raw: str, task: Task) -> GuardVerdict:
        data, error = parse_json_object(raw)
        if error or data is None:
            return _reject(self.name, error or "malformed_json")
        validator = Draft202012Validator(task.json_schema)
        errors = sorted(validator.iter_errors(data), key=lambda item: list(item.path))
        if not errors:
            return _accept(self.name)
        first = errors[0]
        path = ".".join(str(part) for part in first.path)
        detail = first.message if not path else f"{path}: {first.message}"
        return _reject(self.name, f"schema_mismatch: {detail}")


class PydanticGuard:
    """Valida contra o modelo da tarefa, sem coerção de string numérica ou booleana."""

    name = "pydantic"

    def check(self, raw: str, task: Task) -> GuardVerdict:
        data, error = parse_json_object(raw)
        if error or data is None:
            return _reject(self.name, error or "malformed_json")
        model = model_for(task.model_key)
        try:
            model.model_validate(data)
        except ValidationError as exc:
            return _reject(self.name, f"model_mismatch: {_first_model_error(exc)}")
        return _accept(self.name)


class RuleGuard:
    """Aplica só as regras listadas na tarefa. Campo extra não é, por si, falha."""

    name = "rules"

    def check(self, raw: str, task: Task) -> GuardVerdict:
        data, error = parse_json_object(raw)
        if error or data is None:
            return _reject(self.name, error or "malformed_json")
        for rule in task.rules:
            reason = apply_rule(data, rule)
            if reason:
                return _reject(self.name, reason)
        return _accept(self.name)


class SecondLLMGuard:
    """Pede um veredito JSON ao revisor. Resposta fora do contrato rejeita (falha fechada)."""

    name = "second_llm"

    def __init__(self, checker: LLMClient) -> None:
        self._checker = checker

    def check(self, raw: str, task: Task) -> GuardVerdict:
        response = self._checker.complete(
            [
                {"role": "system", "content": CHECKER_SYSTEM},
                {"role": "user", "content": checker_user(task, raw)},
            ]
        )
        usage_kwargs: dict[str, Any]
        if response.usage is None:
            usage_kwargs = {"extra_call_without_usage": True}
        else:
            usage_kwargs = {"extra_usage": response.usage}
        parsed, error = parse_json_object(response.text)
        if error or parsed is None:
            return _reject(self.name, "checker_malformed", **usage_kwargs)
        accept = parsed.get("accept")
        reason = parsed.get("reason")
        if not isinstance(accept, bool) or not isinstance(reason, str):
            return _reject(self.name, "checker_malformed", **usage_kwargs)
        if accept:
            return _accept(self.name, **usage_kwargs)
        return _reject(self.name, f"checker_rejected: {reason}", **usage_kwargs)


CHECKER_SYSTEM = (
    "You check whether a candidate JSON extraction is faithful to the source text. "
    'Reply with a JSON object {"accept": true or false, "reason": "short explanation"}. '
    "accept must be a JSON boolean. Do not wrap the JSON in markdown."
)


def checker_user(task: Task, raw: str) -> str:
    # O gabarito não entra no prompt. Incluí-lo tornaria o revisor uma consulta à resposta.
    schema = json.dumps(task.json_schema, ensure_ascii=False, sort_keys=True)
    return (
        f"Instruction:\n{task.instruction}\n\n"
        f"Source:\n{task.source_text}\n\n"
        f"Schema:\n{schema}\n\n"
        f"Candidate:\n{raw}"
    )


def build_guard(strategy: str, *, checker: LLMClient | None = None) -> Guard:
    if strategy == "json_schema":
        return JsonSchemaGuard()
    if strategy == "pydantic":
        return PydanticGuard()
    if strategy == "rules":
        return RuleGuard()
    if strategy == "second_llm":
        if checker is None:
            raise ValueError("second_llm strategy requires a checker LLM")
        return SecondLLMGuard(checker)
    raise ValueError(f"unknown strategy: {strategy}")


def _first_model_error(exc: ValidationError) -> str:
    first = exc.errors()[0]
    loc = ".".join(str(part) for part in first.get("loc", ()))
    message = str(first.get("msg", "invalid"))
    if not loc:
        return message
    return f"{loc}: {message}"


def apply_rule(data: dict[str, Any], rule: Rule) -> str | None:
    try:
        handler = _HANDLERS[rule.op]
    except KeyError as exc:
        raise ValueError(f"unknown rule op: {rule.op}") from exc
    return handler(data, rule)


def _missing(field: str) -> str:
    return f"rule_failed: missing {field}"


def _required_number(value: float | None) -> float:
    if value is None:
        raise ValueError("rule is missing value")
    return value


def _rule_regex(data: dict[str, Any], rule: Rule) -> str | None:
    field = rule.field or ""
    if field not in data:
        return _missing(field)
    value = data[field]
    if not isinstance(value, str):
        return f"rule_failed: {field} is not a string"
    if re.fullmatch(rule.pattern or "", value) is None:
        return f"rule_failed: {field} does not match {rule.pattern}"
    return None


def _rule_type(data: dict[str, Any], rule: Rule) -> str | None:
    field = rule.field or ""
    if field not in data:
        return _missing(field)
    if _matches_json_type(data[field], rule.json_type or ""):
        return None
    return f"rule_failed: {field} is not {rule.json_type}"


def _rule_gte(data: dict[str, Any], rule: Rule) -> str | None:
    field = rule.field or ""
    if field not in data:
        return _missing(field)
    if rule.numeric not in {"integer", "number"}:
        raise ValueError(f"gte numeric must be integer or number, got {rule.numeric}")
    expected = "integer" if rule.numeric == "integer" else "number"
    if not _matches_json_type(data[field], expected):
        return f"rule_failed: {field} is not {expected}"
    bound = _required_number(rule.value)
    if data[field] < bound:
        return f"rule_failed: {field} < {bound}"
    return None


def _rule_calendar(data: dict[str, Any], rule: Rule) -> str | None:
    field = rule.field or ""
    if field not in data:
        return _missing(field)
    value = data[field]
    if not isinstance(value, str):
        return f"rule_failed: {field} is not a date string"
    try:
        datetime.date.fromisoformat(value)
    except ValueError:
        return f"rule_failed: {field} is not a calendar date"
    return None


def _rule_enum(data: dict[str, Any], rule: Rule) -> str | None:
    field = rule.field or ""
    if field not in data:
        return _missing(field)
    if data[field] not in (rule.choices or []):
        return f"rule_failed: {field} not in enum"
    return None


def _rule_max_when_eq(data: dict[str, Any], rule: Rule) -> str | None:
    # Condição falsa não inspeciona o campo: a regra é condicional, não um required.
    when_field = rule.when_field or ""
    if data.get(when_field) != rule.when_value:
        return None
    field = rule.field or ""
    if field not in data or not _matches_json_type(data[field], "integer"):
        return f"rule_failed: {field} is not an integer"
    bound = _required_number(rule.value)
    if data[field] > bound:
        return f"rule_failed: {field} above {bound} when {when_field}={rule.when_value}"
    return None


def _rule_product(data: dict[str, Any], rule: Rule) -> str | None:
    names = list(rule.factors or [])
    product_field = rule.product or ""
    numbers: list[int] = []
    for name in names:
        if name not in data:
            return _missing(name)
        if not _matches_json_type(data[name], "integer"):
            return f"rule_failed: {name} is not an integer"
        numbers.append(data[name])
    if product_field not in data:
        return _missing(product_field)
    if not _matches_json_type(data[product_field], "integer"):
        return f"rule_failed: {product_field} is not an integer"
    expected = 1
    for number in numbers:
        expected *= number
    if expected != data[product_field]:
        rendered = "*".join(names)
        return f"rule_failed: product_equals {rendered} != {product_field}"
    return None


def _rule_text_length(data: dict[str, Any], rule: Rule) -> str | None:
    field = rule.field or ""
    if field not in data:
        return _missing(field)
    value = data[field]
    if isinstance(value, str):
        targets: list[object] = [value]
    elif isinstance(value, list):
        targets = list(value)
    else:
        return f"rule_failed: {field} is not text"
    low = rule.min_length or 0
    high = rule.max_length or 0
    for index, item in enumerate(targets):
        if not isinstance(item, str):
            return f"rule_failed: {field}[{index}] is not text"
        if not low <= len(item) <= high:
            return f"rule_failed: {field} length out of range"
    return None


def _rule_list_len(data: dict[str, Any], rule: Rule) -> str | None:
    field = rule.field or ""
    if field not in data:
        return _missing(field)
    value = data[field]
    if not isinstance(value, list):
        return f"rule_failed: {field} is not a list"
    if not (rule.minimum or 0) <= len(value) <= (rule.maximum or 0):
        return f"rule_failed: {field} list length out of range"
    return None


def _matches_json_type(value: object, json_type: str) -> bool:
    if json_type == "string":
        return isinstance(value, str)
    if json_type == "boolean":
        return isinstance(value, bool)
    if json_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if json_type == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    raise ValueError(f"unknown json_type: {json_type}")


_HANDLERS = {
    "regex": _rule_regex,
    "gte": _rule_gte,
    "type": _rule_type,
    "calendar_date": _rule_calendar,
    "enum": _rule_enum,
    "max_when_eq": _rule_max_when_eq,
    "product_equals": _rule_product,
    "text_length": _rule_text_length,
    "list_len": _rule_list_len,
}
