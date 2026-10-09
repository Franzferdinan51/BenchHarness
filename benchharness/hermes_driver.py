"""Hermes driver: run HermesBench tasks (am423/hermes-bench-tool-call)
through the real Hermes Agent, routed to local LM Studio.

Task contract (verified live 2026-10-09 against hermes-bench v0.3.0):
- tasks live in <repo>/tasks/<group>/<task>/task.yaml, id "<group>/<task>"
- one trial: `python -m hermesbench.run_real --model <id>
  --base-url <lm-studio> --tasks <id> --run-id <rid>
  --hermes-agent-path <checkout>` (cwd = repo root; hermesbench needs
  only yaml + click on this path, both already in our venv)
- local no-auth servers: OPENAI_API_KEY=dummy keeps run_agent.py on
  the supplied --base-url (upstream's own documented routing)
- verdict: results/<rid>/summary.json -> tasks[0]
  {status: PASS|FAIL|..., score, reason, elapsed_seconds}
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

REPO_URL = "https://github.com/am423/hermes-bench-tool-call"
REPO_ENV_VAR = "HERMESBENCH_PATH"
AGENT_ENV_VAR = "HERMES_AGENT_PATH"


def default_repo_dir() -> Path:
    """Checkout location (override with $HERMESBENCH_PATH)."""
    override = os.environ.get(REPO_ENV_VAR, "").strip()
    if override:
        return Path(override)
    return Path.home() / ".cache" / "benchharness" / "hermes-bench-tool-call"


def default_agent_path() -> Path:
    """Hermes Agent checkout (override with $HERMES_AGENT_PATH)."""
    override = os.environ.get(AGENT_ENV_VAR, "").strip()
    if override:
        return Path(override)
    return Path.home() / ".hermes" / "hermes-agent"


def ensure_repo(timeout_secs: float = 300.0) -> Path:
    """Clone (shallow) or fast-forward the hermes-bench checkout."""
    from benchharness.sandbox import run_local

    dest = default_repo_dir()
    if dest.is_dir() and (dest / "tasks").is_dir():
        pulled = run_local(
            ["git", "-C", str(dest), "pull", "--ff-only"],
            timeout_secs=timeout_secs,
        )
        if pulled.exit_code == 0:
            return dest
        # Offline or diverged: use the cached checkout as-is.
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    proc = run_local(
        ["git", "clone", "--depth", "1", REPO_URL, str(dest)],
        timeout_secs=timeout_secs,
    )
    if proc.exit_code != 0 or not (dest / "tasks").is_dir():
        raise RuntimeError(f"hermes-bench clone failed: {proc.stderr[-400:]}")
    return dest


def require_agent_checkout() -> Path:
    """Hermes Agent checkout containing run_agent.py (raises if missing)."""
    path = default_agent_path()
    if not (path / "run_agent.py").is_file():
        raise RuntimeError(
            f"Hermes Agent checkout not found at {path} "
            f"(need run_agent.py; set ${AGENT_ENV_VAR})"
        )
    return path


def list_tasks(repo_dir: Path) -> list[str]:
    """Task ids (<group>/<task>) for every task.yaml under tasks/."""
    tasks_root = repo_dir / "tasks"
    if not tasks_root.is_dir():
        return []
    out = []
    for path in sorted(tasks_root.glob("t*/t*/task.yaml")):
        rel = path.parent.relative_to(tasks_root)
        if rel.parts[0].startswith("_"):
            continue
        out.append(f"{rel.parts[0]}/{rel.parts[1]}")
    return out


@dataclass
class HermesOutcome:
    task_id: str
    passed: bool
    score: float
    reason: str = ""
    seconds: float = 0.0
    error: str = ""


def parse_summary(summary_path: Path) -> HermesOutcome | None:
    """Read results/<run>/summary.json into an outcome (None if absent)."""
    if not summary_path.is_file():
        return None
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    tasks = data.get("tasks") or []
    if not tasks:
        return None
    row = tasks[0]
    status = str(row.get("status", "")).upper()
    return HermesOutcome(
        task_id=str(row.get("task_id", "")),
        passed=status == "PASS",
        score=float(row.get("score", 1.0 if status == "PASS" else 0.0)),
        reason=str(row.get("reason", "")),
        seconds=float(row.get("elapsed_seconds", 0.0) or 0.0),
    )


def run_task(
    task_id: str,
    model: str,
    base_url: str,
    repo_dir: Path | None = None,
    agent_path: Path | None = None,
    run_id: str | None = None,
    timeout_secs: float = 1200.0,
) -> tuple[HermesOutcome | None, str]:
    """Run one hermes-bench task; returns (outcome, tail)."""
    from benchharness.sandbox import run_local

    repo = repo_dir or ensure_repo()
    agent = agent_path or require_agent_checkout()
    rid = run_id or f"bh-{uuid.uuid4().hex[:8]}"
    env = {**os.environ, "OPENAI_API_KEY": "dummy"}
    proc = run_local(
        [
            sys.executable,
            "-m",
            "hermesbench.run_real",
            "--repo-root",
            str(repo),
            "--model",
            model,
            "--base-url",
            base_url,
            "--tasks",
            task_id,
            "--run-id",
            rid,
            "--hermes-agent-path",
            str(agent),
        ],
        cwd=repo,
        timeout_secs=timeout_secs,
        env=env,
    )
    tail = (proc.stdout + "\n" + proc.stderr)[-2000:].strip()
    outcome = parse_summary(repo / "results" / rid / "summary.json")
    if outcome is None and proc.timed_out:
        outcome = HermesOutcome(
            task_id=task_id,
            passed=False,
            score=0.0,
            error=f"timeout after {timeout_secs:.0f}s",
        )
    return outcome, tail
