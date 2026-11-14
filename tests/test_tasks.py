"""Contrato da fixture placeholder: gabarito aceito e subtle só errado no fato."""

from __future__ import annotations

import json

import pytest

from guardrails_study.guards import JsonSchemaGuard, PydanticGuard, RuleGuard
from guardrails_study.tasks import load_tasks, script_from_labels

_TASKS = load_tasks()


def test_placeholder_has_between_8_and_12_tasks():
    assert 8 <= len(_TASKS) <= 12


def test_task_ids_are_unique():
    ids = [task.id for task in _TASKS]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("task", _TASKS, ids=lambda task: task.id)
def test_ground_truth_passes_deterministic_guards(task):
    raw = json.dumps(task.ground_truth)
    assert JsonSchemaGuard().check(raw, task).accepted
    assert PydanticGuard().check(raw, task).accepted
    assert RuleGuard().check(raw, task).accepted


@pytest.mark.parametrize("task", _TASKS, ids=lambda task: task.id)
def test_malformed_examples_fail_json_and_deterministic_guards(task):
    with pytest.raises(json.JSONDecodeError):
        json.loads(task.examples.malformed)
    assert not JsonSchemaGuard().check(task.examples.malformed, task).accepted
    assert not PydanticGuard().check(task.examples.malformed, task).accepted
    assert not RuleGuard().check(task.examples.malformed, task).accepted


@pytest.mark.parametrize("task", _TASKS, ids=lambda task: task.id)
def test_subtle_examples_differ_from_truth_and_pass_deterministic_guards(task):
    parsed = json.loads(task.examples.subtle)
    assert parsed != task.ground_truth
    assert JsonSchemaGuard().check(task.examples.subtle, task).accepted
    assert PydanticGuard().check(task.examples.subtle, task).accepted
    assert RuleGuard().check(task.examples.subtle, task).accepted


def test_script_from_labels_emits_the_three_classes():
    task = _TASKS[0]
    valid, malformed, subtle = script_from_labels(task, ["valid", "malformed", "subtle"])
    assert json.loads(valid) == task.ground_truth
    with pytest.raises(json.JSONDecodeError):
        json.loads(malformed)
    assert json.loads(subtle) != task.ground_truth


def test_script_from_labels_rejects_unknown_label():
    with pytest.raises(ValueError, match="unknown script label"):
        script_from_labels(_TASKS[0], ["nope"])  # type: ignore[list-item]
