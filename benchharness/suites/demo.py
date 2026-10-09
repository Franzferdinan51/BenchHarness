"""Golden demo suite: 2 deterministic tasks for smoke tests and CI."""

from __future__ import annotations

from benchharness.schema import Score
from benchharness.suites.base import SuiteAdapter, Task, strip_thinking


class DemoAdapter(SuiteAdapter):
    name = "demo"
    category = "reasoning"
    description = "Golden demo suite (2 deterministic tasks, no deps)"
    status = "wired"

    def tasks(self, limit=None):
        tasks = [
            Task(task_id="echo-1", prompt='Reply with exactly: PINEAPPLE',
                 reference="PINEAPPLE"),
            Task(task_id="math-1", prompt='Reply with exactly: 42',
                 reference="42"),
        ]
        return tasks[:limit] if limit else tasks

    def score(self, output, task):
        ok = task.reference in strip_thinking(output)
        return Score(passed=ok, details=f"contains({task.reference!r})={ok}")
