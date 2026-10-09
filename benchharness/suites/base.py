"""Adapter contract. Every benchmark suite implements SuiteAdapter:

- ``name`` / ``category`` / ``description`` identify it for the registry.
- ``requirements()`` lists pip packages / CLIs / docker images; missing ones
  make the runner SKIP tasks with a clear reason instead of failing.
- ``tasks(limit)`` yields Task objects (prompt-ready).
- ``messages(task)`` builds the chat prompt.
- ``score(output, task)`` grades the model text.
- ``prepare(workdir)`` is an optional per-run setup hook (clone repos, etc).
"""

from __future__ import annotations

import re
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from benchharness.schema import Score

_THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove <think>/<thinking> reasoning blocks (Ornith/Qwen3-style).

    Scorers must grade the final answer, not the trace: traces contain
    decoy letters, fake diffs, and exploratory wrong answers.
    """
    cleaned = _THINK_RE.sub("", text)
    # Unterminated block (truncated at cap): drop from the opener on.
    cleaned = re.sub(r"<think(?:ing)?>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


@dataclass
class Task:
    task_id: str
    prompt: str
    reference: str = ""
    metadata: dict = field(default_factory=dict)
    system: str = ""


@dataclass
class Requirement:
    kind: str  # "pip" | "cli" | "docker" | "env" | "note" | "evalscope"
    name: str
    detail: str = ""
    soft: bool = False  # soft: degraded mode (e.g. smoke samples), don't skip


def _requirement_missing(req: Requirement) -> bool:
    if req.kind == "pip":
        return not _have_module(req.name)
    if req.kind == "cli":
        return shutil.which(req.name) is None
    if req.kind == "env":
        return not _have_env(req.name)
    if req.kind == "evalscope":
        from benchharness.evalscope_driver import evalscope_python
        return not evalscope_python().is_file()
    return False  # docker/note checked at execution time or informational


class SuiteAdapter(ABC):
    name: str = ""
    category: str = ""  # coding | reasoning | agentic
    description: str = ""
    max_tokens: int = 2048

    def requirements(self) -> list[Requirement]:
        return []

    def missing_requirements(self, hard_only: bool = False) -> list[Requirement]:
        missing: list[Requirement] = []
        for req in self.requirements():
            if hard_only and req.soft:
                continue
            if _requirement_missing(req):
                missing.append(req)
        return missing

    def prepare(self, workdir: Path) -> None:
        """Optional per-run setup (download data, build images). No-op default."""

    def score_with_client(self, output: str, task: Task, client, model: str) -> Score | None:
        """LLM-judge grading hook. Return a Score to use the judge verdict,
        or None to keep the synchronous score(). Only called when judging
        is enabled (BENCH_JUDGE=1)."""
        return None

    def run_external(self, task: Task, ctx: dict) -> tuple[str, Score] | None:
        """External execution hook (Harbor/docker loops).

        Return (output_excerpt, Score) to bypass the chat flow, or None to
        use the standard prompt -> chat -> score path. ctx carries model,
        config, workdir, run_id.
        """
        return None

    @abstractmethod
    def tasks(self, limit: int | None = None) -> list[Task]:
        ...

    def messages(self, task: Task) -> list[dict]:
        msgs: list[dict] = []
        if task.system:
            msgs.append({"role": "system", "content": task.system})
        msgs.append({"role": "user", "content": task.prompt})
        return msgs

    @abstractmethod
    def score(self, output: str, task: Task) -> Score:
        ...


def _have_module(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


def _have_env(name: str) -> bool:
    import os

    return bool(os.environ.get(name, "").strip())
