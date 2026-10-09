"""EvalScope driver: run Claw-Eval / DeepSWE tasks through EvalScope's
official runners against an OpenAI-compatible endpoint (LM Studio).

EvalScope + claw-eval live in an isolated venv (their dependency tree is
heavy); this module shells out to that interpreter, mirroring harbor_driver.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


def evalscope_python() -> Path:
    """Interpreter of the isolated EvalScope venv (override via env)."""
    override = os.environ.get("BENCH_EVALSCOPE_PYTHON", "").strip()
    if override:
        return Path(override)
    return Path.home() / ".cache" / "benchharness" / "evalscope-venv" / "bin" / "python"


def have_evalscope(benchmark: str = "claw_eval") -> bool:
    """True when the venv python exists and imports evalscope (+extras)."""
    py = evalscope_python()
    if not (py.is_file() and os.access(py, os.X_OK)):
        return False
    module = {"claw_eval": "claw_eval", "deep_swe": "pier"}.get(benchmark, "")
    script = "import evalscope" + (f", {module}" if module else "")
    try:
        proc = subprocess.run([str(py), "-c", script], capture_output=True,
                              timeout=120.0)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


@dataclass
class EvalScopeOutcome:
    task_id: str
    score: float
    passed: bool
    details: str
    trace_path: str = ""
    error: str = ""


DRIVER_SCRIPT = r"""
import glob, json, os, sys

benchmark, task_id, split, model, api_base, api_key, trials, work_dir = sys.argv[1:9]
trials = int(trials)

from evalscope import run_task
from evalscope.config import TaskConfig

dataset_args = {benchmark: {"extra_params": {"task_ids": [task_id]}}}
if benchmark == "claw_eval":
    dataset_args[benchmark]["subset_list"] = [split]
if benchmark == "deep_swe":
    dataset_args[benchmark]["extra_params"]["pier_agent_kwargs"] = {"model_class": "litellm"}

cfg = TaskConfig(
    model=model,
    api_url=api_base,
    api_key=api_key,
    datasets=[benchmark],
    dataset_args=dataset_args,
    judge={"strategy": "llm", "models": [
        {"model_id": model, "api_url": api_base, "api_key": api_key}]},
    repeats=trials,
    work_dir=work_dir,
)
run_task(task_cfg=cfg)

# One task per run: every review row belongs to it (repeats -> N rows).
rows = []
patterns = [os.path.join(work_dir, "*", "reviews", "*", "*.jsonl"),
            os.path.join(work_dir, "outputs", "*", "reviews", "*", "*.jsonl")]
seen_paths = set()
for pattern in patterns:
    for path in sorted(glob.glob(pattern)):
        if path in seen_paths:
            continue
        seen_paths.add(path)
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                sc = (row.get("sample_score") or {}).get("score") or {}
                rows.append({"value": sc.get("value") or {},
                             "status": sc.get("status") or "",
                             "metadata": sc.get("metadata") or {}})
print("BENCH_SAMPLES_JSON:" + json.dumps(rows, default=str))
"""


def run_one(
    benchmark: str,
    task_id: str,
    *,
    model: str,
    api_base: str,
    api_key: str,
    split: str = "general",
    trials: int = 1,
    work_dir: Path | None = None,
    timeout_secs: float = 3600.0,
) -> EvalScopeOutcome:
    """Run one benchmark task via EvalScope; parse the returned report."""
    import uuid as _uuid

    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in task_id)[:60]
    dest = (work_dir or Path.cwd()) / f"evalscope-{benchmark}-{safe}-{_uuid.uuid4().hex[:6]}"
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [str(evalscope_python()), "-c", DRIVER_SCRIPT, benchmark, task_id,
           split, model, api_base, api_key, str(trials), str(dest)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout_secs)
    except subprocess.TimeoutExpired:
        return EvalScopeOutcome(task_id, 0.0, False,
                                f"evalscope timeout after {timeout_secs:.0f}s",
                                error="timeout")
    except OSError as exc:
        return EvalScopeOutcome(task_id, 0.0, False, f"launch failed: {exc}",
                                error="launch")
    samples: list = []
    for line in (proc.stdout or "").splitlines():
        if line.startswith("BENCH_SAMPLES_JSON:"):
            try:
                samples = json.loads(line[len("BENCH_SAMPLES_JSON:"):])
            except ValueError:
                samples = []
    if not samples:
        tail = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()[-800:]
        return EvalScopeOutcome(task_id, 0.0, False,
                                f"no samples (rc={proc.returncode}): {tail}",
                                error="no-result")
    return parse_samples(task_id, samples)


def _num(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def parse_samples(task_id: str, samples: list[dict]) -> EvalScopeOutcome:
    """Build one outcome from review rows (one row per repeat trial).

    Metric shapes (confirmed by stub-endpoint probe):
    - claw_eval value: {task_score, passed, error_rate, judge_score}
    - deep_swe value: {acc} with metadata {reward, f2p, p2p, ...}
    Pass requires EVERY trial to pass (official Pass^3 when trials=3).
    """
    verdicts: list[bool] = []
    scores: list[float] = []
    trace_path = ""
    errors: list[str] = []
    for sample in samples:
        value = sample.get("value") or {}
        meta = sample.get("metadata") or {}
        if "passed" in value:
            verdicts.append(_num(value["passed"]) >= 1.0)
            scores.append(_num(value.get("task_score", value["passed"])))
        elif "acc" in value:
            verdicts.append(_num(value["acc"]) >= 1.0)
            scores.append(_num(value["acc"]))
        else:
            first = next(iter(value.values()), 0.0)
            scores.append(_num(first))
            verdicts.append(_num(first) >= 1.0)
        trace_path = trace_path or str(
            meta.get("trace_path") or meta.get("pier_job_result_path") or "")
        err = meta.get("error") or ""
        if err and err not in errors:
            errors.append(str(err)[:200])
    if not verdicts:
        return EvalScopeOutcome(task_id, 0.0, False, "empty samples",
                                error="parse")
    mean = sum(scores) / len(scores)
    passed = all(verdicts)
    details = f"score={mean:.3f} trials={len(verdicts)}"
    if errors:
        details += f" error={errors[0]}"
    return EvalScopeOutcome(task_id, mean, passed, details,
                            trace_path=trace_path)
