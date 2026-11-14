"""Harness offline: roda o dataset e grava registro por tarefa mais o resumo.

O backend padrão é o fake. Número produzido por ele é checagem de encanamento,
não achado sobre modelo. Provedor ao vivo só segue com --allow-live e chave no
ambiente; sem os dois o processo sai com código 2 e não envia pedido.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from guardrails_study.config import Settings, build_llm
from guardrails_study.dataset import EvalTask, eval_dataset_path, load_eval_tasks
from guardrails_study.llm import LLMClient, LLMResponse, ScriptedLLM
from guardrails_study.metrics import (
    SCHEMA_VERSION,
    STRATEGY_ORDER,
    Observation,
    Summary,
    build_summary,
    classify,
    output_matches_ground_truth,
)
from guardrails_study.runner import AttemptRecord, attempts_for, run_strategy
from guardrails_study.tasks import ScriptLabel, Task, script_from_labels

RECORDS_FILENAME = "records.jsonl"
SUMMARY_FILENAME = "summary.json"
EXIT_USAGE = 1
EXIT_LIVE_GATE = 2

SCENARIOS: tuple[str, ...] = ("always_valid", "malformed_then_valid", "subtle")

PIPELINE_NOTE = "Scripted backend output is a pipeline check, not a model finding."
LIVE_NOTE = (
    "Live provider run. Metrics compare the guard decision to the ground truth outside the guard."
)

_CANDIDATE_MARKER = "\n\nCandidate:\n"
_ERROR_LIMIT = 500
_REPO_ROOT = Path(__file__).resolve().parents[1]


class _StudyParser(argparse.ArgumentParser):
    """Uso inválido sai com código 1. Código 2 fica reservado à recusa do provedor ao vivo."""

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        print(f"error: {message}", file=sys.stderr)
        raise SystemExit(EXIT_USAGE)


class EvalRecord(BaseModel):
    """Linha de records.jsonl. valid é o aceite da guarda, como no RunResult."""

    model_config = ConfigDict(extra="forbid")
    schema_version: int
    task_id: str
    category: str
    strategy: str
    scenario: str | None
    backend: str
    status: Literal["ok", "error"]
    error: str | None
    final_output: str | None
    valid: bool | None
    matches_ground_truth: bool | None
    escaped_invalid: bool | None
    false_rejection: bool | None
    budget_exhausted: bool | None
    attempts: int | None
    max_attempts: int | None
    rejected_by: str | None
    latency_ms: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    cost_usd: float | None
    history: list[AttemptRecord]

    def observation(self) -> Observation:
        return Observation(
            strategy=self.strategy,
            category=self.category,
            status=self.status,
            valid=self.valid,
            matches_ground_truth=self.matches_ground_truth,
            attempts=self.attempts,
            max_attempts=self.max_attempts,
            latency_ms=self.latency_ms,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            cost_usd=self.cost_usd,
        )


class CanonicalChecker:
    """Revisor do fake: aceita só o objeto igual ao gabarito.

    A guarda real não recebe o gabarito. Este duplo não é um modelo; o motivo
    devolvido não cita a resposta, para o reparo não virar consulta ao rótulo.
    """

    def __init__(self, ground_truth: Mapping[str, object]) -> None:
        self._ground_truth = ground_truth

    def complete(self, messages: list[dict[str, str]], *, model: str | None = None) -> LLMResponse:
        raw = _candidate_text(messages)
        accept = output_matches_ground_truth(raw, self._ground_truth)
        reason = "faithful to the source" if accept else "not faithful to the source"
        body = json.dumps({"accept": accept, "reason": reason})
        return LLMResponse(text=body, usage=None, model=model)


def script_for_scenario(task: Task, scenario: str, slots: int) -> list[str]:
    """Monta a fila do fake com o tamanho do orçamento, para ela não reciclar."""
    if slots < 1:
        raise ValueError("slots must be >= 1")
    labels = _scenario_labels(scenario, slots)
    return script_from_labels(task, labels)


def evaluate_one(
    task: EvalTask,
    strategy: str,
    settings: Settings,
    generator: LLMClient,
    *,
    checker: LLMClient | None,
    scenario: str | None,
    backend: str,
) -> EvalRecord:
    """Executa uma tarefa. Exceção de transporte vira status error, não rejeição."""
    budget = attempts_for(strategy, settings)
    try:
        result = run_strategy(strategy, task, generator, settings, checker=checker)
    except Exception as exc:
        return _error_record(task, strategy, scenario, backend, budget, exc)
    matches = output_matches_ground_truth(result.final_output, task.ground_truth)
    flags = classify(
        valid=result.valid,
        matches_ground_truth=matches,
        attempts=result.attempts,
        max_attempts=budget,
    )
    return EvalRecord(
        schema_version=SCHEMA_VERSION,
        task_id=task.id,
        category=task.category,
        strategy=strategy,
        scenario=scenario,
        backend=backend,
        status="ok",
        error=None,
        final_output=result.final_output,
        valid=result.valid,
        matches_ground_truth=matches,
        escaped_invalid=flags.escaped_invalid,
        false_rejection=flags.false_rejection,
        budget_exhausted=flags.budget_exhausted,
        attempts=result.attempts,
        max_attempts=budget,
        rejected_by=result.rejected_by,
        latency_ms=result.latency_ms,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        cost_usd=result.cost_usd,
        history=list(result.history),
    )


def run_evaluation(
    tasks: Sequence[EvalTask],
    strategies: Sequence[str],
    settings: Settings,
    *,
    scenario: str | None,
    backend: str,
    dataset: str,
    note: str,
) -> tuple[list[EvalRecord], Summary]:
    """Uma fila nova por tarefa no fake. O cliente ao vivo, se houver, é compartilhado."""
    live_generator, live_checker = _live_clients(settings, strategies, backend)
    records: list[EvalRecord] = []
    for task in tasks:
        for strategy in strategies:
            generator, checker = _clients_for(
                task,
                strategy,
                settings,
                backend=backend,
                scenario=scenario,
                live_generator=live_generator,
                live_checker=live_checker,
            )
            records.append(
                evaluate_one(
                    task,
                    strategy,
                    settings,
                    generator,
                    checker=checker,
                    scenario=scenario,
                    backend=backend,
                )
            )
    summary = build_summary(
        [record.observation() for record in records],
        dataset=dataset,
        backend=backend,
        scenario=scenario,
        strategies=strategies,
        max_attempts=settings.max_attempts,
        note=note,
    )
    return records, summary


def write_outputs(output_dir: Path, records: Sequence[EvalRecord], summary: Summary) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record.model_dump(mode="json"), ensure_ascii=False) for record in records]
    body = ("\n".join(lines) + "\n") if lines else ""
    (output_dir / RECORDS_FILENAME).write_text(body, encoding="utf-8")
    payload = json.dumps(summary.model_dump(mode="json"), indent=2, ensure_ascii=False)
    (output_dir / SUMMARY_FILENAME).write_text(payload + "\n", encoding="utf-8")


def live_refusals(
    settings: Settings,
    strategies: Sequence[str],
    *,
    allow_live: bool,
) -> list[str]:
    """Recusa provedor ao vivo antes de montar cliente. A mensagem não leva segredo."""
    messages: list[str] = []
    generator = _role_refusal(
        settings.llm_provider,
        settings.llm_model,
        settings.llm_api_key,
        role="generator",
        allow_live=allow_live,
    )
    if generator is not None:
        messages.append(generator)
    if "second_llm" not in strategies:
        return messages
    # O fake não mistura com revisor ao vivo: ou os dois são scriptados, ou os dois são ao vivo.
    if settings.llm_provider == "fake" and settings.checker_provider != "fake":
        messages.append(
            f"refusing live provider {settings.checker_provider} for checker: "
            "the fake generator does not call a live checker. "
            "Use a live LLM_PROVIDER with --allow-live for a live checker. "
            "No request was sent."
        )
        return messages
    if settings.llm_provider != "fake" and settings.checker_provider == "fake":
        messages.append(
            "refusing second_llm: CHECKER_PROVIDER is fake while the generator is live. "
            "No request was sent."
        )
        return messages
    checker = _role_refusal(
        settings.checker_provider,
        settings.checker_model,
        settings.checker_api_key,
        role="checker",
        allow_live=allow_live,
    )
    if checker is not None:
        messages.append(checker)
    return messages


def dataset_label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(_REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None, *, env: Mapping[str, str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    source = os.environ if env is None else env
    try:
        settings = Settings.from_env(source)
    except ValueError as exc:
        return _fail(str(exc))
    if args.provider is not None:
        settings = settings.model_copy(update={"llm_provider": args.provider})
    if args.max_attempts is not None:
        if args.max_attempts < 1:
            return _fail("max-attempts must be >= 1")
        settings = settings.model_copy(update={"max_attempts": args.max_attempts})
    strategies = list(STRATEGY_ORDER) if args.strategy == "all" else [args.strategy]
    refusals = live_refusals(settings, strategies, allow_live=args.allow_live)
    if refusals:
        for message in refusals:
            print(f"error: {message}", file=sys.stderr)
        return EXIT_LIVE_GATE
    backend = settings.llm_provider
    if backend != "fake" and args.scenario is not None:
        return _fail("scenario applies only to the fake backend. No request was sent.")
    scenario = None if backend != "fake" else (args.scenario or "always_valid")
    if args.output.exists() and not args.output.is_dir():
        return _fail(f"output path is not a directory: {args.output}")
    dataset_path = eval_dataset_path() if args.tasks is None else args.tasks
    try:
        tasks = load_eval_tasks(dataset_path)
    except ValueError as exc:
        return _fail(str(exc))
    note = PIPELINE_NOTE if backend == "fake" else LIVE_NOTE
    try:
        records, summary = run_evaluation(
            tasks,
            strategies,
            settings,
            scenario=scenario,
            backend=backend,
            dataset=dataset_label(dataset_path),
            note=note,
        )
    except ValueError as exc:
        return _fail(f"{exc}. No request was sent.")
    write_outputs(args.output, records, summary)
    errors = sum(record.status == "error" for record in records)
    print(
        f"wrote {len(records)} records ({errors} errors) to {args.output}",
        file=sys.stderr,
    )
    if backend == "fake":
        print(PIPELINE_NOTE, file=sys.stderr)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = _StudyParser(
        prog="evaluate-guards",
        description=(
            "Run guard strategies over the eval set and write records.jsonl plus summary.json. "
            "The default backend is the scripted fake. Its numbers are a pipeline check, "
            "not a model finding."
        ),
    )
    parser.add_argument(
        "--strategy",
        required=True,
        choices=[*STRATEGY_ORDER, "all"],
        help="Guard strategy, or all four.",
    )
    parser.add_argument(
        "--scenario",
        choices=SCENARIOS,
        default=None,
        help=(
            "Scripted generator queue. Default for the fake backend: always_valid. "
            "Not valid with a live provider."
        ),
    )
    parser.add_argument("--output", type=Path, required=True, help="Directory for the two files.")
    parser.add_argument(
        "--tasks",
        type=Path,
        default=None,
        help="Eval JSONL. Default: data/eval/tasks.v1.jsonl.",
    )
    parser.add_argument(
        "--provider",
        choices=["fake", "openai", "anthropic"],
        default=None,
        help="Generator provider. Default: LLM_PROVIDER, or fake when unset.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Retry budget for pydantic, rules and second_llm. Default: MAX_ATTEMPTS or 3.",
    )
    parser.add_argument(
        "--allow-live",
        action="store_true",
        help=(
            "Required before openai or anthropic. Without it the process exits 2 "
            "and does not send a request."
        ),
    )
    return parser


def _scenario_labels(scenario: str, slots: int) -> list[ScriptLabel]:
    if scenario == "always_valid":
        return ["valid"] * slots
    if scenario == "malformed_then_valid":
        tail: list[ScriptLabel] = ["valid"] * (slots - 1)
        return ["malformed", *tail]
    if scenario == "subtle":
        return ["subtle"] * slots
    raise ValueError(f"unknown scenario: {scenario}")


def _live_clients(
    settings: Settings,
    strategies: Sequence[str],
    backend: str,
) -> tuple[LLMClient | None, LLMClient | None]:
    # O import do SDK acontece no primeiro complete, não aqui. Sem --allow-live isto não roda.
    if backend == "fake":
        return None, None
    generator = build_llm(settings, role="generator")
    checker = build_llm(settings, role="checker") if "second_llm" in strategies else None
    return generator, checker


def _clients_for(
    task: EvalTask,
    strategy: str,
    settings: Settings,
    *,
    backend: str,
    scenario: str | None,
    live_generator: LLMClient | None,
    live_checker: LLMClient | None,
) -> tuple[LLMClient, LLMClient | None]:
    if backend == "fake":
        if scenario is None:
            raise ValueError("fake backend requires a scenario")
        slots = attempts_for(strategy, settings)
        checker = CanonicalChecker(task.ground_truth) if strategy == "second_llm" else None
        return ScriptedLLM(script_for_scenario(task, scenario, slots)), checker
    if live_generator is None:
        raise ValueError("live backend requires a generator client")
    checker = live_checker if strategy == "second_llm" else None
    return live_generator, checker


def _role_refusal(
    provider: str,
    model: str,
    api_key: str,
    *,
    role: str,
    allow_live: bool,
) -> str | None:
    if provider == "fake":
        return None
    if not allow_live:
        return (
            f"refusing live provider {provider} for {role}: "
            "--allow-live is required. No request was sent."
        )
    missing: list[str] = []
    model_var = "LLM_MODEL" if role == "generator" else "CHECKER_MODEL"
    key_var = "LLM_API_KEY" if role == "generator" else "CHECKER_API_KEY"
    if not model:
        missing.append(model_var)
    if not api_key:
        missing.append(key_var)
    if not missing:
        return None
    verb = "are" if len(missing) > 1 else "is"
    joined = " and ".join(missing)
    return (
        f"refusing live provider {provider} for {role}: {joined} {verb} empty. No request was sent."
    )


def _candidate_text(messages: list[dict[str, str]]) -> str:
    if not messages:
        return ""
    content = messages[-1].get("content", "")
    index = content.rfind(_CANDIDATE_MARKER)
    if index == -1:
        return content
    return content[index + len(_CANDIDATE_MARKER) :]


def _error_text(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}"
    if len(text) <= _ERROR_LIMIT:
        return text
    return text[: _ERROR_LIMIT - 3] + "..."


def _error_record(
    task: EvalTask,
    strategy: str,
    scenario: str | None,
    backend: str,
    budget: int,
    exc: Exception,
) -> EvalRecord:
    return EvalRecord(
        schema_version=SCHEMA_VERSION,
        task_id=task.id,
        category=task.category,
        strategy=strategy,
        scenario=scenario,
        backend=backend,
        status="error",
        error=_error_text(exc),
        final_output=None,
        valid=None,
        matches_ground_truth=None,
        escaped_invalid=None,
        false_rejection=None,
        budget_exhausted=None,
        attempts=None,
        max_attempts=budget,
        rejected_by=None,
        latency_ms=None,
        prompt_tokens=None,
        completion_tokens=None,
        cost_usd=None,
        history=[],
    )


def _fail(message: str) -> int:
    print(f"error: {message}", file=sys.stderr)
    return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
