"""Runner: executes suites against the LM client with concurrency,
per-task timeouts, tool loops, JSONL streaming, and resume support.
"""

from __future__ import annotations

import re
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

from benchharness.config import BenchConfig
from benchharness.lm_client import LMStudioClient
from benchharness.registry import get_suite
from benchharness.sandbox import run_python_snippet
from benchharness.schema import RunSummary, TaskResult, append_result, read_results
from benchharness.suites.base import SuiteAdapter, Task

TOOL_FENCE = re.compile(r"```tool\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def run_tool_loop(
    client: LMStudioClient,
    adapter: SuiteAdapter,
    task: Task,
    model: str,
    max_tokens: int,
) -> tuple[str, int, int]:
    """hle-tools style loop: ```tool fences execute run_python, up to N rounds."""
    rounds = int(task.metadata.get("tool_rounds", 3))
    messages = adapter.messages(task)
    prompt_tokens = completion_tokens = 0
    output = ""
    for _ in range(rounds + 1):
        resp = client.chat(messages, model=model, max_tokens=max_tokens)
        output = client.extract_text(resp)
        pt, ct = client.extract_usage(resp)
        prompt_tokens += pt
        completion_tokens += ct
        m = TOOL_FENCE.search(output)
        if not m:
            return output, prompt_tokens, completion_tokens
        code = m.group(1).strip()
        result = run_python_snippet(code)
        messages.append({"role": "assistant", "content": output})
        messages.append({
            "role": "user",
            "content": f"tool(run_python) exit={result.exit_code}\n"
                       f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}\n"
                       "Continue: either use the tool again or give the final answer.",
        })
    return output, prompt_tokens, completion_tokens


def effective_max_tokens(config: BenchConfig | None, adapter: SuiteAdapter) -> int:
    if config is not None and config.max_tokens_override:
        return config.max_tokens_override
    return adapter.max_tokens


def evaluate_task(
    client: LMStudioClient,
    adapter: SuiteAdapter,
    task: Task,
    model: str,
    run_id: str,
    config: BenchConfig | None = None,
) -> TaskResult:
    if task.metadata.get("skip_reason"):
        return TaskResult(run_id=run_id, model=model, suite=adapter.name,
                          task_id=task.task_id, passed=False, status="skipped",
                          error=str(task.metadata["skip_reason"]))
    started = time.monotonic()
    cap = effective_max_tokens(config, adapter)
    try:
        finish = ""
        if task.metadata.get("tool_loop") == "run_python":
            output, pt, ct = run_tool_loop(client, adapter, task, model, cap)
        else:
            resp = client.chat(adapter.messages(task), model=model, max_tokens=cap)
            output = client.extract_text(resp)
            pt, ct = client.extract_usage(resp)
            finish = client.extract_finish_reason(resp)
        score = adapter.score(output, task)
        details = score.details
        if finish and finish != "stop":
            details = f"{details} [finish={finish}]".strip()
        return TaskResult(
            run_id=run_id, model=model, suite=adapter.name, task_id=task.task_id,
            passed=score.passed, score=score.score,
            latency_ms=int((time.monotonic() - started) * 1000),
            prompt_tokens=pt, completion_tokens=ct,
            details=details,
            output_excerpt=output[:500],
        )
    except Exception as exc:  # per-task isolation: record, don't crash the run
        return TaskResult(
            run_id=run_id, model=model, suite=adapter.name, task_id=task.task_id,
            passed=False, status="error", error=f"{type(exc).__name__}: {exc}",
            latency_ms=int((time.monotonic() - started) * 1000),
        )


def run_suites(
    suite_names: list[str],
    config: BenchConfig,
    limit: int | None = None,
    resume_from: Path | None = None,
    progress_cb=None,
) -> tuple[Path, RunSummary]:
    """Run suites; returns (run_dir, summary). Streams results to JSONL."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    run_dir = config.out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"

    done: set[tuple[str, str]] = set()
    prior: list[TaskResult] = []
    if resume_from is not None:
        prior = read_results(resume_from / "results.jsonl")
        done = {(r.suite, r.task_id) for r in prior if r.status == "done"}
        for r in prior:
            append_result(results_path, r)

    started_at = datetime.now(timezone.utc).isoformat()
    work: list[tuple[SuiteAdapter, Task]] = []
    skipped: list[TaskResult] = []
    for name in suite_names:
        adapter = get_suite(name)
        missing = adapter.missing_requirements(hard_only=True)
        if missing:
            reason = "missing: " + ", ".join(f"{m.kind}:{m.name}" for m in missing)
            skipped.append(TaskResult(run_id=run_id, model=config.model or "?",
                                      suite=name, task_id="*",
                                      passed=False, status="skipped", error=reason))
            continue
        try:
            adapter.prepare(run_dir / "work" / name)
            tasks = adapter.tasks(limit)
        except Exception as exc:
            skipped.append(TaskResult(
                run_id=run_id, model=config.model or "?", suite=name,
                task_id="*", passed=False, status="error",
                error=f"prepare/tasks failed: {type(exc).__name__}: {exc}"))
            continue
        for task in tasks:
            if (name, task.task_id) not in done:
                work.append((adapter, task))

    results: list[TaskResult] = list(prior) + list(skipped)
    for r in skipped:
        append_result(results_path, r)

    with LMStudioClient(config) as client:
        model = config.model or client.resolve_model()
        for r in results:
            if r.model == "?":
                r.model = model

        def _one(item: tuple[SuiteAdapter, Task]) -> TaskResult:
            adapter, task = item
            res = evaluate_task(client, adapter, task, model, run_id, config)
            if progress_cb:
                progress_cb(res)
            return res

        # httpx.Client is thread-safe for distinct requests; bound the pool.
        # Each task gets config.per_task_timeout_secs wall-clock (covers all
        # LM retries); overruns are recorded as timeout errors, not hangs.
        pending: dict = {}
        pool = ThreadPoolExecutor(max_workers=max(1, config.jobs))
        timed_out = False
        try:
            for item in work:
                pending[pool.submit(_one, item)] = item
            while pending:
                done, _ = wait(list(pending), timeout=config.per_task_timeout_secs,
                               return_when=FIRST_COMPLETED)
                if not done:
                    timed_out = True
                    for fut, (adapter, task) in list(pending.items()):
                        fut.cancel()
                        res = TaskResult(
                            run_id=run_id, model=model, suite=adapter.name,
                            task_id=task.task_id, passed=False, status="error",
                            error=f"timeout after {config.per_task_timeout_secs:g}s",
                        )
                        results.append(res)
                        append_result(results_path, res)
                    pending.clear()
                    break
                for fut in done:
                    results.append(fut.result())
                    append_result(results_path, results[-1])
                    del pending[fut]
        finally:
            # On timeout, don't let straggler threads hold the run hostage;
            # each is still bounded by the httpx request timeout + retries.
            pool.shutdown(wait=not timed_out, cancel_futures=timed_out)

    summary = RunSummary.from_results(run_id, model, suite_names, started_at, results)
    summary.write(run_dir)
    return run_dir, summary
