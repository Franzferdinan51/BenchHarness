"""Interactive Textual TUI: pick suites, configure, run, watch, inspect.

Launch with `bench-harness tui` (needs the `tui` extra: textual).
All execution goes through benchharness.runner.run_suites in a worker
thread; the TUI never loads/unloads models, it only routes to whatever
LM Studio already has loaded (same contract as the CLI).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

try:
    from textual import on, work
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical
    from textual.message import Message
    from textual.screen import ModalScreen
    from textual.widgets import (
        Button,
        Checkbox,
        DataTable,
        Footer,
        Header,
        Input,
        Label,
        ProgressBar,
        RichLog,
        Select,
        SelectionList,
        Static,
    )
    from textual.widgets.selection_list import Selection
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "the TUI needs the 'tui' extra: pip install 'benchharness[tui]'"
    ) from exc

from benchharness.config import BenchConfig
from benchharness.registry import REGISTRY, list_suites
from benchharness.schema import TaskResult, read_results


@dataclass
class TUIResult(Message):
    """One finished task, posted from the runner thread."""

    result: TaskResult


@dataclass
class TUIFinished(Message):
    """Run completed: (run_dir, summary dict, error)."""

    run_dir: str
    summary: dict
    error: str = ""


@dataclass
class TUIModels(Message):
    """Loaded-model list fetched from LM Studio (or error)."""

    models: list[str]
    resolved: str
    error: str = ""


class ResultsScreen(ModalScreen[None]):
    """Browse one finished run: summary + per-task table."""

    BINDINGS: ClassVar[list] = [
        ("escape", "dismiss", "Close"),
        ("q", "dismiss", "Close"),
    ]

    def __init__(self, run_dir: Path) -> None:
        super().__init__()
        self.run_dir = run_dir

    def compose(self) -> ComposeResult:
        yield Label(f"Run: {self.run_dir}", id="results-title")
        yield Static("", id="results-summary")
        table = DataTable(id="results-table")
        table.add_column("suite/task", width=42)
        table.add_column("status", width=10)
        table.add_column("score", width=8)
        table.add_column("ms", width=10)
        table.add_column("detail")
        yield table
        yield Button("Close", id="results-close")

    def on_mount(self) -> None:
        import json

        summary_path = self.run_dir / "summary.json"
        summary: dict[str, Any] = {}
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except ValueError:
                summary = {}
        model = summary.get("model", "?")
        total = summary.get("total", 0)
        passed = summary.get("passed", 0)
        errors = summary.get("errors", 0)
        skipped = summary.get("skipped", 0)
        self.query_one("#results-summary", Static).update(
            f"model={model} total={total} passed={passed} "
            f"errors={errors} skipped={skipped}"
        )
        table = self.query_one("#results-table", DataTable)
        try:
            results = read_results(self.run_dir / "results.jsonl")
        except OSError:
            results = []
        for r in results:
            status = (
                r.status if r.status != "done" else ("PASS" if r.passed else "fail")
            )
            table.add_row(
                f"{r.suite}/{r.task_id}",
                status,
                f"{r.score:.2f}",
                str(r.latency_ms),
                (r.details or r.error)[:100],
            )

    @on(Button.Pressed, "#results-close")
    def close_dialog(self) -> None:
        self.dismiss(None)


class BenchApp(App[None]):
    """Full interactive benchmark console."""

    TITLE = "BenchHarness"
    SUB_TITLE = "local LLM benchmarks via LM Studio"
    CSS = """
    #main { height: 1fr; }
    #left { width: 38%; border-right: solid $primary; }
    #right { width: 62%; }
    #config { height: auto; padding: 1; border-bottom: solid $primary; }
    #buttons { height: 3; padding: 0 1; }
    #status { height: 1; padding: 0 1; }
    #log { height: 1fr; border-top: solid $primary; }
    #progress { height: 1; padding: 0 1; }
    .row { height: 3; }
    .row Label { width: 16; }
    .row Input, .row Select { width: 1fr; }
    """

    BINDINGS: ClassVar[list] = [
        ("r", "run", "Run"),
        ("s", "stop", "Stop"),
        ("d", "doctor", "Doctor"),
        ("q", "quit", "Quit"),
    ]

    def __init__(
        self, out_dir: Path | None = None, fetch_on_mount: bool = True
    ) -> None:
        super().__init__()
        self.out_dir = out_dir or BenchConfig.load().out_dir
        self._fetch_on_mount = fetch_on_mount
        self._benchmark_running = False
        self._done = 0
        self._pt = 0
        self._ct = 0
        self._current_run_dir: Path | None = None
        self._stop = threading.Event()

    # -- layout ------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            with Vertical(id="left"):
                yield Label("Suites (space to toggle)")
                yield SelectionList[str](id="suites")
                yield Label("Past runs (enter to inspect)")
                yield SelectionList[str](id="runs")
            with Vertical(id="right"):
                with Vertical(id="config"):
                    with Horizontal(classes="row"):
                        yield Label("Model")
                        yield Select[str](
                            [("auto (first loaded)", "")], id="model", value=""
                        )
                    with Horizontal(classes="row"):
                        yield Label("Workers")
                        yield Input("4", id="jobs")
                    with Horizontal(classes="row"):
                        yield Label("Limit/suite")
                        yield Input("", placeholder="all", id="limit")
                    with Horizontal(classes="row"):
                        yield Label("Task filter")
                        yield Input("", placeholder="substring", id="task")
                    with Horizontal(classes="row"):
                        yield Label("Out dir")
                        yield Input(str(self.out_dir), id="outdir")
                    with Horizontal(classes="row"):
                        yield Checkbox("Sequential", id="sequential")
                        yield Checkbox("LLM judge", id="judge")
                with Horizontal(id="buttons"):
                    yield Button("Run [r]", id="run", variant="primary")
                    yield Button("Stop [s]", id="stop")
                    yield Button("Doctor [d]", id="doctor")
                    yield Button("Refresh runs", id="refresh")
                yield Static("idle", id="status")
                yield ProgressBar(id="progress", show_eta=False)
                yield RichLog(id="log", highlight=True, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        suites = self.query_one("#suites", SelectionList)
        for s in list_suites():
            suites.add_option(
                Selection(f"{s.name} [{s.category}]", s.name, initial_state=False)
            )
        self.refresh_runs()
        if self._fetch_on_mount:
            self.fetch_models()
        self.query_one("#log", RichLog).write(
            "[bold]BenchHarness TUI[/bold] — select suites, then Run."
        )

    # -- LM Studio models --------------------------------------------

    @work(thread=True, exclusive=True)
    def fetch_models(self) -> None:
        from benchharness.lm_client import LMStudioClient

        cfg = BenchConfig.load()
        try:
            with LMStudioClient(cfg) as client:
                info = client.ping()
                models = info.get("models", []) if info.get("ok") else []
                resolved = client.resolve_model() if models else ""
        except Exception as exc:
            self.post_message(TUIModels([], "", error=str(exc)))
            return
        self.post_message(TUIModels(models, resolved))

    @on(TUIModels)
    def update_models(self, msg: TUIModels) -> None:
        import os as _os

        log = self.query_one("#log", RichLog)
        select = self.query_one("#model", Select)
        if msg.error:
            log.write(f"[red]LM Studio unreachable:[/red] {msg.error}")
            self.query_one("#status", Static).update("LM Studio unreachable")
            return
        # Auto label names the resolution source, mirroring
        # local-grok-cli's env -> first-loaded chain.
        source = ""
        for var in ("LM_STUDIO_MODEL", "GROK_MODEL"):
            if _os.environ.get(var, "").strip():
                source = f"{var}={_os.environ[var].strip()}"
                break
        if not source and msg.resolved:
            source = f"first loaded: {msg.resolved}"
        auto_label = f"auto ({source})" if source else "auto"
        options = [(auto_label, "")]
        options += [(m, m) for m in msg.models]
        select.set_options(options)
        self._endpoint = BenchConfig.load().base_url
        self._resolved_model = msg.resolved
        self._render_status()
        log.write(f"[green]LM Studio ok[/green] — resolved: {msg.resolved}")

    def _render_status(self) -> None:
        endpoint = getattr(self, "_endpoint", "?")
        model = getattr(self, "_resolved_model", "?")
        extra = ""
        if self._benchmark_running:
            extra = f" — done {self._done} • {self._pt}+{self._ct} tok"
        self.query_one("#status", Static).update(f"{endpoint} | {model}{extra}")

    # -- past runs ----------------------------------------------------

    def refresh_runs(self) -> None:
        runs = self.query_one("#runs", SelectionList)
        runs.clear_options()
        out = Path(self.query_one("#outdir", Input).value or str(self.out_dir))
        if not out.is_dir():
            return
        found = [
            run_dir
            for suite_dir in sorted(out.iterdir())
            if suite_dir.is_dir()
            for run_dir in sorted(suite_dir.iterdir())
            if (run_dir / "summary.json").is_file()
        ]
        for run_dir in found[-30:]:
            rel = run_dir.relative_to(out)
            runs.add_option(Selection(str(rel), str(run_dir)))

    @on(SelectionList.SelectedChanged, "#runs")
    def inspect_run(self, event: SelectionList.SelectedChanged) -> None:
        if event.selection:
            self.push_screen(ResultsScreen(Path(event.selection)))

    @on(Button.Pressed, "#refresh")
    def on_refresh(self) -> None:
        self.refresh_runs()

    # -- run control --------------------------------------------------

    def selected_suites(self) -> list[str]:
        return list(self.query_one("#suites", SelectionList).selected)

    def action_run(self) -> None:
        if self._benchmark_running:
            return
        names = self.selected_suites()
        if not names:
            self.query_one("#log", RichLog).write(
                "[yellow]Select at least one suite first.[/yellow]"
            )
            return
        try:
            jobs = int(self.query_one("#jobs", Input).value or "4")
            if jobs < 1:
                raise ValueError
        except ValueError:
            self.query_one("#log", RichLog).write(
                "[red]Workers must be an integer >= 1.[/red]"
            )
            return
        limit_raw = self.query_one("#limit", Input).value.strip()
        try:
            limit = int(limit_raw) if limit_raw else None
        except ValueError:
            self.query_one("#log", RichLog).write(
                "[red]Limit must be blank or an int.[/red]"
            )
            return
        if self.query_one("#sequential", Checkbox).value:
            jobs = 1
        task_raw = self.query_one("#task", Input).value.strip()
        model = self.query_one("#model", Select).value or None
        if model == "":
            model = None
        cfg = BenchConfig.load(
            {
                "model": model,
                "jobs": jobs,
                "judge_enabled": self.query_one("#judge", Checkbox).value or None,
                "out_dir": Path(self.query_one("#outdir", Input).value or "."),
            }
        )
        self._benchmark_running = True
        self._done = 0
        self._pt = 0
        self._ct = 0
        self._stop.clear()
        self.query_one("#progress", ProgressBar).update(total=None)
        self.query_one("#log", RichLog).write(
            f"running {len(names)} suite(s), {jobs} worker(s)..."
        )
        self._render_status()
        self.run_benchmark(names, cfg, limit, [task_raw] if task_raw else None)

    @work(thread=True, exclusive=True)
    def run_benchmark(
        self,
        names: list[str],
        cfg: BenchConfig,
        limit: int | None,
        task_filter: list[str] | None,
    ) -> None:
        from benchharness.runner import run_suites

        def _progress(res: TaskResult) -> None:
            self.post_message(TUIResult(res))

        try:
            run_dir, summary = run_suites(
                names,
                cfg,
                limit=limit,
                progress_cb=_progress,
                task_filter=task_filter,
                stop_event=self._stop,
            )
            payload = {
                "total": summary.total,
                "passed": summary.passed,
                "errors": summary.errors,
                "skipped": summary.skipped,
                "pass_at_1": summary.pass_at_1,
                "mean": summary.mean_score,
                "model": summary.model,
            }
            self.post_message(TUIFinished(str(run_dir), payload))
        except Exception as exc:
            self.post_message(TUIFinished("", {}, error=str(exc)))

    @on(TUIResult)
    def show_result(self, msg: TUIResult) -> None:
        res = msg.result
        self._done += 1
        self._pt += res.prompt_tokens or 0
        self._ct += res.completion_tokens or 0
        mark = (
            "PASS"
            if res.passed
            else (
                "SKIP"
                if res.status == "skipped"
                else ("ERR" if res.status == "error" else "fail")
            )
        )
        color = (
            "green" if res.passed else ("yellow" if res.status == "skipped" else "red")
        )
        self.query_one("#log", RichLog).write(
            f"[{color}]{mark}[/{color}] {res.suite}/{res.task_id} ({res.latency_ms} ms)"
        )
        self._render_status()

    @on(TUIFinished)
    def show_finished(self, msg: TUIFinished) -> None:
        self._benchmark_running = False
        log = self.query_one("#log", RichLog)
        if msg.error:
            log.write(f"[red]run failed:[/red] {msg.error}")
            self.query_one("#status", Static).update("run failed")
            return
        s = msg.summary
        log.write(
            f"[bold]finished:[/bold] total={s['total']} passed={s['passed']} "
            f"errors={s['errors']} skipped={s['skipped']} "
            f"pass@1={s['pass_at_1']:.3f} → {msg.run_dir}"
        )
        self._render_status()
        self._current_run_dir = Path(msg.run_dir)
        self.refresh_runs()

    def action_stop(self) -> None:
        if self._benchmark_running:
            self._stop.set()
            self.query_one("#log", RichLog).write(
                "[yellow]Stopping — unstarted tasks will be marked cancelled.[/yellow]"
            )

    @on(Button.Pressed, "#run")
    def on_run(self) -> None:
        self.action_run()

    @on(Button.Pressed, "#stop")
    def on_stop(self) -> None:
        self.action_stop()

    # -- doctor --------------------------------------------------------

    def action_doctor(self) -> None:
        self.query_one("#status", Static).update("pinging LM Studio...")
        self.fetch_models()

    @on(Button.Pressed, "#doctor")
    def on_doctor(self) -> None:
        self.action_doctor()


def main() -> int:
    """Entry point for `bench-harness tui`."""
    BenchApp().run()
    return 0


def available_suites() -> list[str]:
    """Suite names for tests without launching the app."""
    return sorted(REGISTRY)
