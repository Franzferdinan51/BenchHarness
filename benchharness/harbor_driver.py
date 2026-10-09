"""Harbor driver: runs Harbor-based suites (Terminal-Bench 2.x) as subprocess
jobs and maps Harbor TrialResults onto BenchHarness TaskResults.

Verified against harbor (harbor-framework/harbor, pip-installable):
- datasets: terminal-bench@2.0 in the public registry (89 tasks). 2.1 is
  NOT publicly downloadable as of iteration 5 — the version stays
  configurable (TB_DATASET / --tb-dataset) so 2.1 works when published.
- agents: terminus-2 takes `--ak api_base=<url>` + litellm `-m openai/<id>`
  so it runs against local LM Studio. claude-code shells to the Claude
  Code CLI and needs ANTHROPIC_API_KEY (or an ANTHROPIC_BASE_URL shim).
- results: <jobs-dir>/<job>/trial-*/result.json (TrialResult JSON with
  task_name, verifier_result.rewards, exception_info, timings).

Task enumeration downloads only the dataset manifest (~seconds); docker
images pull on first trial run.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DATASET = "terminal-bench@2.0"
ORACLE_AGENT = "oracle"


def have_harbor() -> bool:
    return shutil.which("harbor") is not None


def default_cache_dir() -> Path:
    return Path(os.environ.get("BENCH_HARBOR_CACHE",
                               Path.home() / ".cache" / "benchharness" / "harbor"))


def dataset_name_version(dataset: str) -> tuple[str, str]:
    if "@" in dataset:
        name, version = dataset.rsplit("@", 1)
        return name, version
    return dataset, ""


def ensure_dataset(dataset: str, cache_dir: Path | None = None,
                   timeout_secs: float = 300.0) -> Path:
    """Download (if needed) and return the local dataset directory."""
    cache = cache_dir or default_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    name, _ = dataset_name_version(dataset)
    dest = cache / name
    if dest.is_dir() and any(dest.iterdir()):
        return dest
    cmd = ["harbor", "download", dataset, "-o", str(cache)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_secs)
    if proc.returncode != 0 or not dest.is_dir():
        raise RuntimeError(f"harbor download {dataset} failed: "
                           f"{proc.stderr[-500:]} {proc.stdout[-500:]}")
    return dest


def list_tasks(dataset_dir: Path) -> list[str]:
    """Task names are the dataset subdirectories containing task.toml."""
    names = sorted(p.name for p in dataset_dir.iterdir()
                   if p.is_dir() and (p / "task.toml").is_file())
    if not names:  # tolerate flat layout: any subdir counts
        names = sorted(p.name for p in dataset_dir.iterdir() if p.is_dir())
    return names


def build_run_command(
    dataset: str,
    agent: str,
    model: str,
    jobs_subdir: Path,
    job_name: str,
    task: str | None = None,
    include_task: str | None = None,
    n_tasks: int | None = None,
    n_concurrent: int = 1,
    agent_kwargs: dict[str, str] | None = None,
    agent_env: dict[str, str] | None = None,
    timeout_multiplier: float = 1.0,
    memory_policy: str | None = None,
) -> list[str]:
    cmd = ["harbor", "run", "-d", dataset, "-a", agent, "-m", model,
           "-o", str(jobs_subdir), "--job-name", job_name,
           "-n", str(n_concurrent), "-q",
           "--timeout-multiplier", str(timeout_multiplier)]
    if memory_policy:
        cmd += ["--memory", memory_policy]
    if task:
        cmd += ["-t", task]
    if include_task:
        cmd += ["-i", include_task]
    if n_tasks:
        cmd += ["-l", str(n_tasks)]
    for key, value in (agent_kwargs or {}).items():
        cmd += ["--ak", f"{key}={value}"]
    for key, value in (agent_env or {}).items():
        cmd += ["--ae", f"{key}={value}"]
    return cmd


@dataclass
class TrialOutcome:
    task_name: str
    passed: bool
    score: float
    rewards: dict = field(default_factory=dict)
    error: str = ""
    seconds: float = 0.0


def _trial_passed(rewards: dict) -> tuple[bool, float]:
    if not rewards:
        return False, 0.0
    try:
        best = max(float(v) for v in rewards.values())
    except (TypeError, ValueError):
        return False, 0.0
    return best >= 1.0 or (0.0 < best <= 1.0 and best >= 0.99), min(1.0, max(0.0, best))


def parse_trial_result(path: Path) -> TrialOutcome | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or "task_name" not in data:
        return None
    rewards = {}
    verifier = data.get("verifier_result") or {}
    if isinstance(verifier, dict) and isinstance(verifier.get("rewards"), dict):
        rewards = dict(verifier["rewards"])
    passed, score = _trial_passed(rewards)
    error = ""
    exc = data.get("exception_info")
    if isinstance(exc, dict):
        error = str(exc.get("exception_type", "")) + ": " + str(exc.get("message", ""))[:300]
    elif data.get("exception_info"):
        error = str(data.get("exception_info"))[:300]
    seconds = 0.0
    for key in ("agent_execution", "verifier", "environment_setup", "agent_setup"):
        timing = data.get(key)
        if isinstance(timing, dict) and isinstance(timing.get("duration"), (int, float)):
            seconds += float(timing["duration"])
    if seconds == 0.0 and data.get("started_at") and data.get("finished_at"):
        try:
            from datetime import datetime
            start = datetime.fromisoformat(str(data["started_at"]).replace("Z", "+00:00"))
            end = datetime.fromisoformat(str(data["finished_at"]).replace("Z", "+00:00"))
            seconds = max(0.0, (end - start).total_seconds())
        except ValueError:
            pass
    return TrialOutcome(task_name=str(data.get("task_name", path.parent.name)),
                        passed=passed and not error, score=score if not error else 0.0,
                        rewards=rewards, error=error, seconds=seconds)


def parse_job_dir(job_dir: Path) -> list[TrialOutcome]:
    """Collect every trial result.json under a harbor job directory."""
    outcomes: list[TrialOutcome] = []
    if not job_dir.is_dir():
        return outcomes
    for path in sorted(job_dir.glob("*/result.json")):
        outcome = parse_trial_result(path)
        if outcome is not None:
            outcomes.append(outcome)
    # Fallback: some versions embed trial_results in the job result.json.
    if not outcomes:
        job_result = job_dir / "result.json"
        if job_result.is_file():
            try:
                data = json.loads(job_result.read_text(encoding="utf-8"))
            except ValueError:
                data = None
            if isinstance(data, dict):
                for item in data.get("trial_results", []) or []:
                    if isinstance(item, dict) and "task_name" in item:
                        rewards = {}
                        verifier = item.get("verifier_result") or {}
                        if isinstance(verifier.get("rewards"), dict):
                            rewards = dict(verifier["rewards"])
                        passed, score = _trial_passed(rewards)
                        outcomes.append(TrialOutcome(
                            task_name=str(item.get("task_name")), passed=passed,
                            score=score, rewards=rewards))
    return outcomes


def run_job(cmd: list[str], timeout_secs: float,
            env: dict[str, str] | None = None) -> tuple[int, str]:
    """Run a harbor job; returns (returncode, tail-of-output). Never raises."""
    started = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout_secs, env=env)
        out = (proc.stdout[-2000:] + "\n" + proc.stderr[-2000:]).strip()
        return proc.returncode, out
    except subprocess.TimeoutExpired as exc:
        out = ""
        if isinstance(exc.stdout, str):
            out += exc.stdout[-2000:]
        if isinstance(exc.stderr, str):
            out += "\n" + exc.stderr[-2000:]
        elapsed = time.monotonic() - started
        return 124, (out.strip() + f"\n[timeout after {elapsed:.0f}s]").strip()
    except OSError as exc:
        return 127, str(exc)
