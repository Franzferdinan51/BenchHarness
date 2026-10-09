"""TUI tests (Textual pilot, sync wrappers) + runner cooperative-cancel."""

from __future__ import annotations

import asyncio
import threading

import pytest

pytest.importorskip("textual")

from benchharness.tui import BenchApp, TUIFinished, TUIModels, TUIResult


def test_stop_event_cancels_pending(tmp_path, mock_config, monkeypatch):
    """Pre-set stop event: all tasks recorded as cancelled-skipped."""
    import httpx

    import benchharness.runner as runner_mod
    from benchharness.runner import run_suites

    real_client = runner_mod.LMStudioClient

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"content": "PINEAPPLE 42"},
                            "finish_reason": "stop",
                        }
                    ]
                },
            )
        return httpx.Response(200, json={"data": []})

    class MockClient(real_client):
        def __init__(self, config):
            super().__init__(config, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(runner_mod, "LMStudioClient", MockClient)
    mock_config.out_dir = tmp_path / "bench-results"
    stop = threading.Event()
    stop.set()
    _, summary = run_suites(["demo"], mock_config, stop_event=stop)
    assert (summary.total, summary.skipped, summary.passed) == (2, 2, 0)


def _run(coro):
    return asyncio.run(coro)


async def _mounts_suites():
    app = BenchApp(fetch_on_mount=False)
    async with app.run_test() as pilot:
        from textual.widgets import SelectionList

        suites = app.query_one("#suites", SelectionList)
        assert len(suites.options) == 20
        await pilot.pause()


def test_app_mounts_suites():
    _run(_mounts_suites())


async def _no_suites_warns():
    app = BenchApp(fetch_on_mount=False)
    async with app.run_test() as pilot:
        await pilot.click("#run")
        await pilot.pause()
        log = app.query_one("#log")
        assert "at least one suite" in log.lines[-1].text


def test_run_without_suites_warns():
    _run(_no_suites_warns())


async def _bad_jobs():
    app = BenchApp(fetch_on_mount=False)
    async with app.run_test() as pilot:
        from textual.widgets import SelectionList

        suites = app.query_one("#suites", SelectionList)
        suites.select(suites.options[0].value)
        app.query_one("#jobs").value = "0"
        await pilot.click("#run")
        await pilot.pause()
        log = app.query_one("#log")
        assert ">= 1" in log.lines[-1].text


def test_invalid_jobs_rejected():
    _run(_bad_jobs())


async def _messages():
    from benchharness.schema import TaskResult

    app = BenchApp(fetch_on_mount=False)
    async with app.run_test() as pilot:
        app.post_message(
            TUIResult(
                TaskResult(
                    run_id="r",
                    model="m",
                    suite="demo",
                    task_id="t1",
                    passed=True,
                    score=1.0,
                    latency_ms=12,
                )
            )
        )
        await pilot.pause()
        log = app.query_one("#log")
        assert "demo/t1" in log.lines[-1].text
        app.post_message(
            TUIFinished(
                "/tmp/r",  # noqa: S108 -- display string only, never created
                {
                    "total": 1,
                    "passed": 1,
                    "errors": 0,
                    "skipped": 0,
                    "pass_at_1": 1.0,
                    "mean": 1.0,
                    "model": "m",
                },
            )
        )
        await pilot.pause()
        assert "finished" in log.lines[-1].text
        assert not app._benchmark_running


def test_result_and_finished_messages():
    _run(_messages())


async def _models_ok():
    app = BenchApp(fetch_on_mount=False)
    async with app.run_test() as pilot:
        app.post_message(TUIModels(["model-a", "model-b"], "model-a"))
        await pilot.pause()
        select = app.query_one("#model")
        values = [value for _, value in select._options]
        assert "model-a" in values and "model-b" in values


def test_models_message_populates_select():
    _run(_models_ok())


async def _models_err():
    app = BenchApp(fetch_on_mount=False)
    async with app.run_test() as pilot:
        app.post_message(TUIModels([], "", error="connection refused"))
        await pilot.pause()
        log = app.query_one("#log")
        assert "unreachable" in log.lines[-1].text


def test_models_error_logged():
    _run(_models_err())
