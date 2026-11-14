"""Atalho de teste para a fixture. Não é superfície pública do estudo."""

from __future__ import annotations

from guardrails_study.tasks import Task, load_tasks, task_by_id


def get_task(task_id: str) -> Task:
    return task_by_id(load_tasks(), task_id)
