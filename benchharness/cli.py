"""CLI: bench-harness list-suites | doctor | run | inspect | export | compare."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from benchharness import __version__
from benchharness.config import BenchConfig
from benchharness.lm_client import LMStudioClient
from benchharness.registry import REGISTRY, list_suites
from benchharness.runner import run_suites
from benchharness.schema import read_results

console = Console()


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer >= 1, got {raw!r}")
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected an integer >= 1, got {value}")
    return value


def resolve_jobs(args: argparse.Namespace) -> int | None:
    """CLI jobs resolution: --sequential forces 1 worker.

    Returns None when neither flag was given (config/env default applies).
    """
    if args.sequential and args.jobs is not None:
        raise RuntimeError("pass either --sequential or --jobs N, not both")
    if args.sequential:
        return 1
    return args.jobs


def cmd_list_suites(args: argparse.Namespace) -> int:
    table = Table(title="BenchHarness suites")
    table.add_column("suite")
    table.add_column("category")
    table.add_column("status")
    table.add_column("description")
    for s in list_suites(args.category):
        table.add_row(s.name, s.category, getattr(s, "status", "?"), s.description)
    console.print(table)
    return 0


def cmd_doctor(_args: argparse.Namespace) -> int:
    cfg = BenchConfig.load()
    console.print(f"[bold]base_url:[/bold] {cfg.base_url}")
    console.print(f"[bold]model:[/bold] {cfg.model or '(auto-pick first loaded)'}")
    with LMStudioClient(cfg) as client:
        try:
            info = client.ping()
        except Exception as exc:
            console.print(f"[red]LM Studio unreachable: {exc}[/red]")
            console.print("hint: start it with `lms server start`, then load a model")
            return 1
        if not info["ok"]:
            console.print("[red]LM Studio reachable but no chat models found[/red]")
            console.print("hint: load a model in LM Studio (llm/vlm, not embeddings)")
            return 1
        console.print(f"[green]ok[/green] {info['model_count']} chat models "
                      f"({info['latency_ms']} ms)")
        for mid in info["models"][:10]:
            console.print(f"  - {mid}")
        try:
            resolved = client.resolve_model()
            console.print(f"[bold]resolved model:[/bold] {resolved}")
        except RuntimeError as exc:
            console.print(f"[yellow]{exc}[/yellow]")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    names: list[str] = []
    for raw in args.suite:
        if raw == "all":
            names.extend(sorted(REGISTRY))
        else:
            names.append(raw)
    # de-dup, preserve order
    names = list(dict.fromkeys(names))
    cfg = BenchConfig.load({
        "model": args.model,
        "jobs": resolve_jobs(args),
        "max_tokens_override": args.max_tokens,
        "judge_enabled": args.judge,
        "out_dir": Path(args.out) if args.out else None,
    })
    mode = "sequential (1 worker)" if cfg.jobs == 1 else f"parallel ({cfg.jobs} workers)"
    console.print(f"[bold]mode:[/bold] {mode}")

    def _progress(res) -> None:
        mark = "PASS" if res.passed else ("SKIP" if res.status == "skipped"
                                          else ("ERR" if res.status == "error" else "fail"))
        console.print(f"[{mark}] {res.suite}/{res.task_id} "
                      f"({res.latency_ms} ms{tokens(res)})")

    def tokens(res) -> str:
        if res.prompt_tokens or res.completion_tokens:
            return f", {res.prompt_tokens}+{res.completion_tokens} tok"
        return ""

    run_dir, summary = run_suites(names, cfg, limit=args.limit,
                                  resume_from=Path(args.resume) if args.resume else None,
                                  progress_cb=_progress,
                                  task_filter=args.task)
    console.print(f"\n[bold]run dir:[/bold] {run_dir}")
    console.print(f"total={summary.total} passed={summary.passed} "
                  f"errors={summary.errors} skipped={summary.skipped} "
                  f"pass@1={summary.pass_at_1:.3f} mean={summary.mean_score:.3f}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    run_dir = Path(args.run)
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        console.print_json(summary_path.read_text(encoding="utf-8"))
    results = read_results(run_dir / "results.jsonl")
    table = Table(title=f"{run_dir.name} ({len(results)} results)")
    table.add_column("suite/task")
    table.add_column("status")
    table.add_column("score")
    table.add_column("ms")
    table.add_column("detail")
    for r in results[: args.limit or len(results)]:
        status = r.status if r.status != "done" else ("PASS" if r.passed else "fail")
        table.add_row(f"{r.suite}/{r.task_id}", status, f"{r.score:.2f}",
                      str(r.latency_ms), (r.details or r.error)[:80])
    console.print(table)
    return 0


def summarize_run(run_dir: Path) -> dict:
    """Per-suite aggregates for one run dir: {model, run_id, suites}."""
    results = read_results(run_dir / "results.jsonl")
    suites: dict[str, dict] = {}
    for r in results:
        agg = suites.setdefault(r.suite, {"total": 0, "passed": 0,
                                          "errors": 0, "skipped": 0,
                                          "score_sum": 0.0})
        if r.status == "skipped":
            agg["skipped"] += 1
        elif r.status == "error":
            agg["errors"] += 1
        else:
            agg["total"] += 1
            agg["passed"] += 1 if r.passed else 0
            agg["score_sum"] += r.score or 0.0
    for agg in suites.values():
        n = agg["total"]
        agg["pass_at_1"] = (agg["passed"] / n) if n else 0.0
        agg["mean"] = (agg["score_sum"] / n) if n else 0.0
    model = results[0].model if results else "?"
    return {"model": model, "run_id": run_dir.name, "suites": suites}


def cmd_compare(args: argparse.Namespace) -> int:
    runs = [summarize_run(Path(d)) for d in args.runs]
    names = [f"{r['model']} ({r['run_id'][-6:]})" for r in runs]
    suite_names = sorted({s for r in runs for s in r["suites"]})
    table = Table(title="run comparison (pass@1, done tasks only)")
    table.add_column("suite")
    for n in names:
        table.add_column(n)
    if len(runs) == 2:
        table.add_column("delta (b-a)")

    def cell(agg):
        if agg is None:
            return "-"
        return f"{agg['pass_at_1']:.3f} ({agg['passed']}/{agg['total']})"

    for suite in suite_names:
        aggs = [r["suites"].get(suite) for r in runs]
        row = [suite] + [cell(a) for a in aggs]
        if len(runs) == 2:
            a, b = aggs
            if a and b and a["total"] and b["total"]:
                row.append(f"{b['pass_at_1'] - a['pass_at_1']:+.3f}")
            else:
                row.append("-")
        table.add_row(*row)
    console.print(table)
    for r, n in zip(runs, names):
        errs = sum(a["errors"] for a in r["suites"].values())
        skips = sum(a["skipped"] for a in r["suites"].values())
        console.print(f"{n}: errors={errs} skipped={skips}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    run_dir = Path(args.run)
    results = read_results(run_dir / "results.jsonl")
    if args.format == "json":
        payload = {"run": run_dir.name, "results": [json.loads(r.to_json()) for r in results]}
        text = json.dumps(payload, indent=2)
    else:  # jsonl passthrough
        text = "\n".join(r.to_json() for r in results)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        console.print(f"wrote {args.output}")
    else:
        console.print(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bench-harness", description="Unified benchmark harness")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    ls = sub.add_parser("list-suites", help="list registered suites")
    ls.add_argument("--category", choices=["coding", "reasoning", "agentic"], default=None)
    ls.set_defaults(func=cmd_list_suites)

    d = sub.add_parser("doctor", help="ping LM Studio and resolve the model")
    d.set_defaults(func=cmd_doctor)

    r = sub.add_parser("run", help="run one or more suites")
    r.add_argument("--suite", action="append", required=True,
                   help="suite name or 'all' (repeatable)")
    r.add_argument("--model", default=None, help="override LM_STUDIO_MODEL")
    r.add_argument("--limit", type=int, default=None, help="max tasks per suite")
    r.add_argument("--task", action="append", default=None,
                   help="only run tasks whose id contains this (repeatable)")
    r.add_argument("--jobs", type=_positive_int, default=None, metavar="N",
                   help="parallel task workers (default 4; 1 = sequential; or BENCH_JOBS)")
    r.add_argument("--sequential", action="store_true",
                   help="run tasks one at a time (same as --jobs 1)")
    r.add_argument("--max-tokens", type=int, default=None,
                   help="override per-task completion cap (or BENCH_MAX_TOKENS)")
    r.add_argument("--judge", dest="judge", action="store_true", default=None,
                   help="enable LLM-judge grading where supported (or BENCH_JUDGE=1)")
    r.add_argument("--out", default=None, help="results root (default bench-results/)")
    r.add_argument("--resume", default=None, help="resume a previous run dir")
    r.set_defaults(func=cmd_run)

    i = sub.add_parser("inspect", help="show a finished run")
    i.add_argument("run", help="run directory")
    i.add_argument("--limit", type=int, default=50)
    i.set_defaults(func=cmd_inspect)

    e = sub.add_parser("export", help="export results as json/jsonl")
    e.add_argument("run", help="run directory")
    e.add_argument("--format", choices=["json", "jsonl"], default="json")
    e.add_argument("--output", default=None)
    e.set_defaults(func=cmd_export)

    c = sub.add_parser("compare", help="compare two or more runs per suite")
    c.add_argument("runs", nargs="+", help="run directories")
    c.set_defaults(func=cmd_compare)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        console.print("\n[yellow]interrupted[/yellow]")
        return 130
    except (KeyError, RuntimeError) as exc:
        console.print(f"[red]error: {exc}[/red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
