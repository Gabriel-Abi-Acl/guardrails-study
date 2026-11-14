"""CLI e encanamento offline. Provedor ao vivo para na porta e não abre rede."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

from guardrails_study.config import Settings
from guardrails_study.dataset import EVAL_CATEGORIES, eval_dataset_path, load_eval_tasks
from guardrails_study.evaluate import (
    EXIT_LIVE_GATE,
    PIPELINE_NOTE,
    RECORDS_FILENAME,
    SUMMARY_FILENAME,
    CanonicalChecker,
    EvalRecord,
    evaluate_one,
    live_refusals,
    main,
    run_evaluation,
    script_for_scenario,
)
from guardrails_study.llm import ScriptedLLM, ScriptedOutput, TokenUsage
from guardrails_study.metrics import STRATEGY_ORDER, Summary, aggregate
from guardrails_study.runner import AttemptRecord, run_strategy

ROOT = Path(__file__).resolve().parents[1]


def _env(**overrides: str) -> dict[str, str]:
    base = {"LLM_PROVIDER": "fake"}
    base.update(overrides)
    return base


def _one_task(tmp_path: Path) -> Path:
    line = eval_dataset_path().read_text(encoding="utf-8").splitlines()[0]
    path = tmp_path / "one.jsonl"
    path.write_text(line + "\n", encoding="utf-8")
    return path


def _load_written(output: Path) -> tuple[list[dict], dict]:
    records_path = output / RECORDS_FILENAME
    summary_path = output / SUMMARY_FILENAME
    text = records_path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    records = [json.loads(line) for line in text.splitlines()]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return records, summary


def test_fake_run_writes_the_schema_and_scores_the_whole_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    output = tmp_path / "out"
    code = main(
        ["--strategy", "all", "--scenario", "always_valid", "--output", str(output)],
        env=_env(),
    )
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out == ""
    assert PIPELINE_NOTE in captured.err
    assert "180 records" in captured.err

    records, summary = _load_written(output)
    assert len(records) == 180
    Summary.model_validate(summary)
    assert set(summary) == set(Summary.model_fields)
    assert summary["schema_version"] == 1
    assert summary["dataset"] == "data/eval/tasks.v1.jsonl"
    assert summary["backend"] == "fake"
    assert summary["scenario"] == "always_valid"
    assert summary["strategies"] == list(STRATEGY_ORDER)
    assert summary["note"] == PIPELINE_NOTE
    assert summary["record_count"] == 180
    assert list(summary["by_strategy"]) == list(STRATEGY_ORDER)
    assert list(summary["by_category"]) == list(EVAL_CATEGORIES)

    tasks = {task.id: task for task in load_eval_tasks()}
    counts = Counter(task.category for task in tasks.values())
    assert summary["overall"]["scored_count"] == 180
    assert summary["overall"]["error_count"] == 0
    assert summary["overall"]["acceptance_rate"] == 1
    assert summary["overall"]["escaped_invalid_rate"] == 0
    assert summary["overall"]["false_rejection_rate"] == 0
    assert summary["overall"]["budget_exhaustion_rate"] == 0
    assert summary["overall"]["attempts"]["mean"] == 1
    assert summary["overall"]["attempts"]["p50"] == 1
    assert summary["overall"]["attempts"]["p90"] == 1
    assert summary["overall"]["prompt_tokens"] is None
    assert summary["overall"]["completion_tokens"] is None
    assert summary["overall"]["cost_usd"] is None
    assert summary["overall"]["latency_ms"]["mean"] >= 0

    for category, expected in counts.items():
        per_strategy = summary["by_strategy"]["json_schema"]["by_category"]
        schema_count = per_strategy[category]["scored_count"]
        assert schema_count == expected
        assert summary["by_category"][category]["scored_count"] == expected * len(STRATEGY_ORDER)

    for record in records:
        EvalRecord.model_validate(record)
        assert set(record) == set(EvalRecord.model_fields)
        assert record["status"] == "ok"
        assert record["valid"] is True
        assert record["matches_ground_truth"] is True
        assert record["escaped_invalid"] is False
        assert record["false_rejection"] is False
        assert record["budget_exhausted"] is False
        assert record["attempts"] == 1
        assert record["prompt_tokens"] is None
        assert record["cost_usd"] is None
        assert record["error"] is None
        assert len(record["history"]) == 1
        assert set(record["history"][0]) == set(AttemptRecord.model_fields)
        task = tasks[record["task_id"]]
        assert record["category"] == task.category
        assert json.loads(record["final_output"]) == task.ground_truth


def test_malformed_then_valid_retries_only_when_the_strategy_has_budget():
    tasks = load_eval_tasks()
    _records, summary = run_evaluation(
        tasks,
        ["json_schema", "pydantic"],
        Settings(),
        scenario="malformed_then_valid",
        backend="fake",
        dataset="data/eval/tasks.v1.jsonl",
        note=PIPELINE_NOTE,
    )
    schema = summary.by_strategy["json_schema"].overall
    assert schema.acceptance_rate == 0
    assert schema.escaped_invalid_rate == 0
    assert schema.false_rejection_rate == 0
    assert schema.budget_exhaustion_rate == 1
    assert schema.attempts is not None
    assert schema.attempts.mean == 1

    pydantic = summary.by_strategy["pydantic"].overall
    assert pydantic.acceptance_rate == 1
    assert pydantic.escaped_invalid_rate == 0
    assert pydantic.false_rejection_rate == 0
    assert pydantic.budget_exhaustion_rate == 0
    assert pydantic.attempts is not None
    assert pydantic.attempts.mean == 2


def test_subtle_scenario_escapes_deterministic_guards_and_not_the_scripted_checker():
    # Checagem de encanamento: o revisor scriptado compara com o gabarito. Não é modelo.
    _records, summary = run_evaluation(
        load_eval_tasks(),
        ["json_schema", "rules", "second_llm"],
        Settings(max_attempts=3),
        scenario="subtle",
        backend="fake",
        dataset="data/eval/tasks.v1.jsonl",
        note=PIPELINE_NOTE,
    )
    for name in ("json_schema", "rules"):
        block = summary.by_strategy[name].overall
        assert block.acceptance_rate == 1
        assert block.escaped_invalid_rate == 1
        assert block.false_rejection_rate == 0
        assert block.budget_exhaustion_rate == 0
    checker = summary.by_strategy["second_llm"].overall
    assert checker.acceptance_rate == 0
    assert checker.escaped_invalid_rate == 0
    assert checker.false_rejection_rate == 0
    assert checker.budget_exhaustion_rate == 1
    assert checker.attempts is not None
    assert checker.attempts.mean == 3


def test_scripted_repair_does_not_include_the_label():
    task = load_eval_tasks()[0]
    generator = ScriptedLLM(script_for_scenario(task, "malformed_then_valid", 3))
    checker = CanonicalChecker(task.ground_truth)
    result = run_strategy("second_llm", task, generator, Settings(max_attempts=3), checker=checker)
    assert result.valid is True
    assert result.attempts == 2
    repair = generator.calls[1][-1]["content"]
    assert task.examples.valid not in repair
    assert "ground_truth" not in repair
    assert "not faithful to the source" in repair


def test_script_lengths_follow_the_budget():
    task = load_eval_tasks()[0]
    assert script_for_scenario(task, "malformed_then_valid", 1) == [task.examples.malformed]
    repeated = script_for_scenario(task, "always_valid", 2)
    assert repeated == [task.examples.valid, task.examples.valid]
    subtle = script_for_scenario(task, "subtle", 3)
    assert subtle == [task.examples.subtle, task.examples.subtle, task.examples.subtle]
    with pytest.raises(ValueError, match="unknown scenario"):
        script_for_scenario(task, "nope", 1)
    with pytest.raises(ValueError, match="slots"):
        script_for_scenario(task, "always_valid", 0)


def test_transport_error_is_recorded_and_excluded_from_rates():
    task = load_eval_tasks()[0]
    record = evaluate_one(
        task,
        "json_schema",
        Settings(),
        ScriptedLLM([]),
        checker=None,
        scenario="always_valid",
        backend="fake",
    )
    assert record.status == "error"
    assert record.valid is None
    assert record.matches_ground_truth is None
    assert record.escaped_invalid is None
    assert record.false_rejection is None
    assert record.budget_exhausted is None
    assert record.attempts is None
    assert record.latency_ms is None
    assert record.prompt_tokens is None
    assert record.cost_usd is None
    assert record.final_output is None
    assert record.history == []
    assert record.max_attempts == 1
    assert "empty" in (record.error or "")
    block = aggregate([record.observation()])
    assert block.scored_count == 0
    assert block.error_count == 1
    assert block.acceptance_rate is None


def test_checker_transport_error_is_not_a_rejection():
    class Boom:
        def complete(self, messages: list[dict[str, str]], *, model: str | None = None) -> None:
            raise RuntimeError("connection reset")

    task = load_eval_tasks()[0]
    record = evaluate_one(
        task,
        "second_llm",
        Settings(),
        ScriptedLLM([task.examples.valid]),
        checker=Boom(),
        scenario="always_valid",
        backend="fake",
    )
    assert record.status == "error"
    assert record.valid is None
    assert "connection reset" in (record.error or "")
    assert aggregate([record.observation()]).false_rejection_rate is None


def test_evaluate_one_copies_tokens_and_cost_from_the_runner():
    task = load_eval_tasks()[0]
    generator = ScriptedLLM(
        [
            ScriptedOutput(
                text=task.examples.valid,
                usage=TokenUsage(prompt_tokens=10, completion_tokens=4),
            )
        ]
    )
    settings = Settings(price_prompt_per_million=1_000_000, price_completion_per_million=2_000_000)
    record = evaluate_one(
        task,
        "json_schema",
        settings,
        generator,
        checker=None,
        scenario="always_valid",
        backend="fake",
    )
    assert record.status == "ok"
    assert record.prompt_tokens == 10
    assert record.completion_tokens == 4
    assert record.cost_usd == pytest.approx(18)
    assert record.matches_ground_truth is True
    assert record.escaped_invalid is False


def test_live_provider_without_flag_exits_2_and_builds_no_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("client construction is not allowed")

    monkeypatch.setattr("guardrails_study.evaluate.build_llm", boom)
    output = tmp_path / "out"
    secret = "super-secret-value"
    code = main(
        ["--strategy", "all", "--provider", "openai", "--output", str(output)],
        env=_env(LLM_MODEL="m", LLM_API_KEY=secret),
    )
    err = capsys.readouterr().err
    assert code == EXIT_LIVE_GATE
    assert "--allow-live is required" in err
    assert "No request was sent" in err
    assert secret not in err
    assert not output.exists()


def test_allow_live_without_credentials_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("client construction is not allowed")

    monkeypatch.setattr("guardrails_study.evaluate.build_llm", boom)
    output = tmp_path / "out"
    code = main(
        [
            "--strategy",
            "json_schema",
            "--provider",
            "anthropic",
            "--allow-live",
            "--output",
            str(output),
        ],
        env=_env(LLM_PROVIDER="anthropic"),
    )
    err = capsys.readouterr().err
    assert code == 2
    assert "LLM_MODEL" in err
    assert "LLM_API_KEY" in err
    assert "No request was sent" in err
    assert not output.exists()


def test_live_checker_is_refused_only_when_second_llm_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("client construction is not allowed")

    monkeypatch.setattr("guardrails_study.evaluate.build_llm", boom)
    output = tmp_path / "out"
    secret = "checker-secret-value"
    code = main(
        [
            "--strategy",
            "second_llm",
            "--allow-live",
            "--tasks",
            str(_one_task(tmp_path)),
            "--output",
            str(output),
        ],
        env=_env(CHECKER_PROVIDER="openai", CHECKER_MODEL="m", CHECKER_API_KEY=secret),
    )
    err = capsys.readouterr().err
    assert code == 2
    assert "does not call a live checker" in err
    assert "--allow-live" in err
    assert secret not in err
    assert not output.exists()

    allowed = tmp_path / "schema"
    code = main(
        [
            "--strategy",
            "json_schema",
            "--tasks",
            str(_one_task(tmp_path)),
            "--output",
            str(allowed),
        ],
        env=_env(CHECKER_PROVIDER="openai", CHECKER_API_KEY=secret),
    )
    assert code == 0
    assert (allowed / RECORDS_FILENAME).is_file()
    assert secret not in capsys.readouterr().err


def test_scenario_with_a_live_provider_does_not_send(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("client construction is not allowed")

    monkeypatch.setattr("guardrails_study.evaluate.build_llm", boom)
    output = tmp_path / "out"
    secret = "live-secret-value"
    code = main(
        [
            "--strategy",
            "json_schema",
            "--provider",
            "openai",
            "--allow-live",
            "--scenario",
            "subtle",
            "--output",
            str(output),
        ],
        env=_env(LLM_MODEL="m", LLM_API_KEY=secret),
    )
    err = capsys.readouterr().err
    assert code == 1
    assert "scenario" in err
    assert "No request was sent" in err
    assert secret not in err
    assert not output.exists()


def test_usage_errors_exit_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    output = tmp_path / "out"
    with pytest.raises(SystemExit) as caught:
        main(["--strategy", "stack", "--output", str(output)])
    assert caught.value.code == 1

    target = tmp_path / "file.txt"
    target.write_text("x", encoding="utf-8")
    code = main(["--strategy", "json_schema", "--output", str(target)], env=_env())
    assert code == 1
    assert "not a directory" in capsys.readouterr().err

    code = main(
        ["--strategy", "json_schema", "--max-attempts", "0", "--output", str(output)],
        env=_env(),
    )
    assert code == 1
    code = main(
        ["--strategy", "json_schema", "--output", str(output)],
        env={"LLM_PROVIDER": "azure"},
    )
    assert code == 1
    assert "LLM_PROVIDER" in capsys.readouterr().err
    missing = tmp_path / "missing.jsonl"
    code = main(
        ["--strategy", "json_schema", "--tasks", str(missing), "--output", str(output)],
        env=_env(),
    )
    assert code == 1
    assert not (output / RECORDS_FILENAME).exists()


def test_live_refusal_messages_name_the_role_and_not_the_secret():
    settings = Settings(
        llm_provider="openai",
        llm_model="m",
        llm_api_key="generator-secret",
        checker_provider="anthropic",
        checker_api_key="",
    )
    messages = live_refusals(settings, ["second_llm"], allow_live=False)
    blob = "\n".join(messages)
    assert "generator" in blob
    assert "checker" in blob
    assert "generator-secret" not in blob
    assert live_refusals(settings, ["json_schema"], allow_live=False)[0].startswith(
        "refusing live provider openai for generator"
    )
    mixed = live_refusals(
        Settings(llm_provider="openai", checker_provider="fake"),
        ["second_llm"],
        allow_live=True,
    )
    assert any("CHECKER_PROVIDER is fake" in message for message in mixed)


def test_module_entrypoint_and_console_script(tmp_path: Path):
    env = {key: os.environ[key] for key in ("PATH", "HOME", "LANG", "LC_ALL") if key in os.environ}
    env["LLM_PROVIDER"] = "openai"
    refused = subprocess.run(
        [
            sys.executable,
            "-m",
            "guardrails_study.evaluate",
            "--strategy",
            "json_schema",
            "--output",
            str(tmp_path / "refused"),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert refused.returncode == 2
    assert "--allow-live" in refused.stderr
    assert "No request was sent" in refused.stderr
    assert not (tmp_path / "refused").exists()

    env["LLM_PROVIDER"] = "fake"
    ran = subprocess.run(
        [
            sys.executable,
            "-m",
            "guardrails_study.evaluate",
            "--strategy",
            "json_schema",
            "--scenario",
            "always_valid",
            "--tasks",
            str(_one_task(tmp_path)),
            "--output",
            str(tmp_path / "ran"),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert ran.returncode == 0, ran.stderr
    assert PIPELINE_NOTE in ran.stderr
    written, summary = _load_written(tmp_path / "ran")
    assert len(written) == 1
    assert summary["overall"]["acceptance_rate"] == 1
    assert summary["note"] == PIPELINE_NOTE

    # resolve() sairia do venv: o interpretador é um link para o Python do uv.
    script = Path(sys.executable).parent / "evaluate-guards"
    assert script.is_file()
    help_run = subprocess.run(
        [str(script), "--help"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_run.returncode == 0
    assert "pipeline check" in help_run.stdout


def test_readme_explains_the_metrics_and_the_scripted_caveat():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for snippet in (
        "evaluate-guards",
        "python -m guardrails_study.evaluate",
        "--allow-live",
        "código 2",
        "acceptance_rate",
        "escaped_invalid_rate",
        "false_rejection_rate",
        "budget_exhaustion_rate",
        "checagem de encanamento",
        "não são achado sobre modelo",
        "nearest-rank",
    ):
        assert snippet in readme
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "runs/" in gitignore.splitlines()
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'evaluate-guards = "guardrails_study.evaluate:main"' in project


def test_importing_the_harness_does_not_import_provider_sdks():
    import guardrails_study.evaluate  # noqa: F401

    assert "openai" not in sys.modules
    assert "anthropic" not in sys.modules
