"""Orçamento de tentativas e o registro que os PRs de métrica vão agregar.

Latência é relógio ao redor da chamada e da guarda. Token e custo só entram
quando o provedor e a configuração realmente informam os dois.
"""

from __future__ import annotations

import json
import time

from pydantic import BaseModel, ConfigDict

from guardrails_study.config import Settings
from guardrails_study.guards import Guard, GuardVerdict, build_guard
from guardrails_study.llm import LLMClient, LLMResponse
from guardrails_study.tasks import Task

# json_schema é validação única. As outras estratégias reperguntam até o orçamento.
SINGLE_SHOT_STRATEGIES = frozenset({"json_schema"})
STRATEGIES = frozenset({"json_schema", "pydantic", "rules", "second_llm"})

SYSTEM_PROMPT = (
    "Return only a JSON object that validates against the schema. No markdown and no commentary."
)


class AttemptRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    attempt: int
    raw_output: str
    accepted: bool
    rejected_by: str | None
    reason: str | None
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None


class RunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: str
    strategy: str
    final_output: str
    valid: bool
    attempts: int
    rejected_by: str | None
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    cost_usd: float | None
    history: list[AttemptRecord]


def attempts_for(strategy: str, settings: Settings) -> int:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy: {strategy}")
    if strategy in SINGLE_SHOT_STRATEGIES:
        return 1
    return settings.max_attempts


def initial_messages(task: Task) -> list[dict[str, str]]:
    schema = json.dumps(task.json_schema, ensure_ascii=False, sort_keys=True)
    user = f"{task.instruction}\n\nSource:\n{task.source_text}\n\nJSON Schema:\n{schema}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def repair_message(verdict: GuardVerdict) -> str:
    reason = verdict.reason or verdict.guard_name
    return (
        f"The previous output was rejected by {verdict.guard_name}: {reason}. "
        "Return only a JSON object. Do not add commentary or markdown."
    )


def run_task(
    task: Task,
    generator: LLMClient,
    guard: Guard,
    *,
    strategy: str,
    max_attempts: int,
    settings: Settings | None = None,
) -> RunResult:
    """Repete enquanto a guarda rejeitar e ainda houver tentativa. Não consulta o gabarito."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    active = settings or Settings()
    messages = initial_messages(task)
    history: list[AttemptRecord] = []
    for number in range(1, max_attempts + 1):
        started = time.perf_counter()
        response = generator.complete(messages)
        verdict = guard.check(response.text, task)
        elapsed_ms = (time.perf_counter() - started) * 1000
        prompt_tokens, completion_tokens = _attempt_usage(response, verdict)
        history.append(
            AttemptRecord(
                attempt=number,
                raw_output=response.text,
                accepted=verdict.accepted,
                rejected_by=None if verdict.accepted else verdict.guard_name,
                reason=verdict.reason,
                latency_ms=elapsed_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )
        if verdict.accepted or number == max_attempts:
            break
        messages.append({"role": "assistant", "content": response.text})
        messages.append({"role": "user", "content": repair_message(verdict)})
    return _result(task.id, strategy, history, active)


def run_strategy(
    strategy: str,
    task: Task,
    generator: LLMClient,
    settings: Settings,
    checker: LLMClient | None = None,
) -> RunResult:
    guard = build_guard(strategy, checker=checker)
    return run_task(
        task,
        generator,
        guard,
        strategy=strategy,
        max_attempts=attempts_for(strategy, settings),
        settings=settings,
    )


def _attempt_usage(response: LLMResponse, verdict: GuardVerdict) -> tuple[int | None, int | None]:
    # Uso parcial não vira zero: ou todas as chamadas da tentativa informam, ou o campo fica null.
    if response.usage is None or verdict.extra_call_without_usage:
        return None, None
    prompt = response.usage.prompt_tokens
    completion = response.usage.completion_tokens
    if verdict.extra_usage is not None:
        prompt += verdict.extra_usage.prompt_tokens
        completion += verdict.extra_usage.completion_tokens
    return prompt, completion


def _result(
    task_id: str, strategy: str, history: list[AttemptRecord], settings: Settings
) -> RunResult:
    last = history[-1]
    latency_ms = sum(item.latency_ms for item in history)
    prompt_tokens, completion_tokens, cost = _rollup(history, settings)
    return RunResult(
        task_id=task_id,
        strategy=strategy,
        final_output=last.raw_output,
        valid=last.accepted,
        attempts=len(history),
        rejected_by=None if last.accepted else last.rejected_by,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost,
        history=history,
    )


def _rollup(
    history: list[AttemptRecord],
    settings: Settings,
) -> tuple[int | None, int | None, float | None]:
    if any(item.prompt_tokens is None or item.completion_tokens is None for item in history):
        return None, None, None
    prompt = sum(item.prompt_tokens or 0 for item in history)
    completion = sum(item.completion_tokens or 0 for item in history)
    if settings.price_prompt_per_million is None or settings.price_completion_per_million is None:
        return prompt, completion, None
    cost = (
        prompt * settings.price_prompt_per_million
        + completion * settings.price_completion_per_million
    ) / 1_000_000
    return prompt, completion, cost
