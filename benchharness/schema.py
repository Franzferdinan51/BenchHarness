"""Result schema: per-task JSONL records plus an aggregate summary.json.

Mirrors the synthesis contract:
    RunResult{run_id, model, suite, task_id, pass, score, latency_ms,
              tokens, error?}
plus a summary with pass@1 / mean score for resume-friendly reporting.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class Score:
    """Normalized outcome of grading one model output."""

    passed: bool
    score: float = 0.0  # 0.0..1.0, suite-defined partial credit allowed
    details: str = ""

    def __post_init__(self) -> None:
        self.score = min(1.0, max(0.0, float(self.score)))
        if self.passed and self.score == 0.0:
            self.score = 1.0


@dataclass
class TaskResult:
    """One evaluated task. Serialized as one JSONL line."""

    run_id: str
    model: str
    suite: str
    task_id: str
    passed: bool
    score: float = 0.0
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    status: str = "done"  # done | error | skipped
    error: str = ""
    details: str = ""
    output_excerpt: str = ""  # first ~500 chars of raw model output
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> "TaskResult":
        return cls(**json.loads(line))


@dataclass
class RunSummary:
    """Aggregate over one `run` invocation, written as summary.json."""

    run_id: str
    model: str
    suites: list[str]
    started_at: str
    finished_at: str = ""
    total: int = 0
    passed: int = 0
    errors: int = 0
    skipped: int = 0
    mean_score: float = 0.0
    total_latency_ms: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    per_suite: dict = field(default_factory=dict)

    @classmethod
    def from_results(
        cls, run_id: str, model: str, suites: list[str], started_at: str, results: list[TaskResult]
    ) -> "RunSummary":
        summary = cls(run_id=run_id, model=model, suites=suites, started_at=started_at)
        summary.finished_at = datetime.now(timezone.utc).isoformat()
        per: dict[str, dict] = {}
        for r in results:
            summary.total += 1
            if r.status == "skipped":
                summary.skipped += 1
            elif r.status == "error":
                summary.errors += 1
            elif r.passed:
                summary.passed += 1
            summary.total_latency_ms += r.latency_ms
            summary.total_prompt_tokens += r.prompt_tokens
            summary.total_completion_tokens += r.completion_tokens
            bucket = per.setdefault(r.suite, {"total": 0, "passed": 0, "errors": 0,
                                              "skipped": 0, "score_sum": 0.0})
            bucket["total"] += 1
            bucket["score_sum"] += r.score
            if r.status == "skipped":
                bucket["skipped"] += 1
            elif r.status == "error":
                bucket["errors"] += 1
            elif r.passed:
                bucket["passed"] += 1
        graded = [r for r in results if r.status == "done"]
        summary.mean_score = sum(r.score for r in graded) / len(graded) if graded else 0.0
        for name, bucket in per.items():
            n = bucket["total"] - bucket["skipped"]
            bucket["mean_score"] = bucket["score_sum"] / n if n else 0.0
            bucket["pass_at_1"] = bucket["passed"] / n if n else 0.0
            del bucket["score_sum"]
        summary.per_suite = per
        return summary

    @property
    def pass_at_1(self) -> float:
        graded = self.total - self.skipped
        return self.passed / graded if graded else 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def write(self, directory: Path) -> Path:
        path = directory / "summary.json"
        path.write_text(self.to_json(), encoding="utf-8")
        return path


def append_result(path: Path, result: TaskResult) -> None:
    """Append one result line (crash-safe resume source)."""
    with path.open("a", encoding="utf-8") as fh:
        fh.write(result.to_json() + "\n")


def read_results(path: Path) -> list[TaskResult]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(TaskResult.from_json(line))
    return out
