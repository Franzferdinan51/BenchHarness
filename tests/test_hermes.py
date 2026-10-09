"""Hermes driver tests: task enumeration, summary parsing, adapter wiring."""

from __future__ import annotations

import json


def _repo(tmp_path):
    root = tmp_path / "hb"
    (root / "tasks" / "t01_terminal_smoke" / "t01_echo").mkdir(parents=True)
    (root / "tasks" / "t01_terminal_smoke" / "t01_echo" / "task.yaml").write_text(
        "id: t01_terminal_smoke/t01_echo\n"
    )
    (root / "tasks" / "t02_file_read" / "t01_cat").mkdir(parents=True)
    (root / "tasks" / "t02_file_read" / "t01_cat" / "task.yaml").write_text("id: x\n")
    (root / "tasks" / "_template").mkdir(parents=True)
    return root


def test_list_tasks_skips_template(tmp_path):
    from benchharness.hermes_driver import list_tasks

    assert list_tasks(_repo(tmp_path)) == [
        "t01_terminal_smoke/t01_echo",
        "t02_file_read/t01_cat",
    ]
    assert list_tasks(tmp_path / "nope") == []


def test_parse_summary_pass_fail(tmp_path):
    from benchharness.hermes_driver import parse_summary

    assert parse_summary(tmp_path / "missing.json") is None
    p = tmp_path / "summary.json"
    p.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": "t01_terminal_smoke/t01_echo",
                        "status": "PASS",
                        "score": 1.0,
                        "reason": "ok",
                        "elapsed_seconds": 26.4,
                    }
                ]
            }
        )
    )
    oc = parse_summary(p)
    assert oc.passed and oc.score == 1.0 and oc.seconds == 26.4
    p.write_text(json.dumps({"tasks": [{"task_id": "t", "status": "FAIL"}]}))
    oc = parse_summary(p)
    assert not oc.passed and oc.score == 0.0
    p.write_text(json.dumps({"tasks": []}))
    assert parse_summary(p) is None


def test_run_task_invokes_runner(tmp_path, monkeypatch):
    import benchharness.hermes_driver as driver
    from benchharness.sandbox import ExecResult

    repo = _repo(tmp_path)
    rid = "bh-test"
    (repo / "results" / rid).mkdir(parents=True)
    (repo / "results" / rid / "summary.json").write_text(
        json.dumps({"tasks": [{"task_id": "t", "status": "PASS", "score": 1.0}]})
    )
    agent = tmp_path / "agent"
    agent.mkdir()
    (agent / "run_agent.py").write_text("# agent")
    seen = {}

    def fake_run(argv, cwd=None, timeout_secs=0.0, env=None):
        seen["argv"] = argv
        seen["cwd"] = cwd
        seen["key"] = (env or {}).get("OPENAI_API_KEY")
        return ExecResult(0, "ok", "")

    import benchharness.sandbox as sandbox_mod

    monkeypatch.setattr(sandbox_mod, "run_local", fake_run)
    oc, tail = driver.run_task(
        "t01_terminal_smoke/t01_echo",
        model="m",
        base_url="http://127.0.0.1:1234/v1",
        repo_dir=repo,
        agent_path=agent,
        run_id=rid,
    )
    assert oc.passed and tail == "ok"
    assert "-m" in seen["argv"] and "hermesbench.run_real" in seen["argv"]
    assert seen["cwd"] == repo and seen["key"] == "dummy"


def test_require_agent_checkout_missing(tmp_path, monkeypatch):
    import pytest

    import benchharness.hermes_driver as driver

    monkeypatch.setenv("HERMES_AGENT_PATH", str(tmp_path / "nope"))
    with pytest.raises(RuntimeError, match=r"run_agent\.py"):
        driver.require_agent_checkout()


def test_hermes_adapter_run_external(tmp_path, monkeypatch):
    import benchharness.hermes_driver as driver
    from benchharness.config import BenchConfig
    from benchharness.registry import get_suite
    from benchharness.suites.base import Task

    def fake_run_task(task_id, **kwargs):
        assert task_id == "t01_terminal_smoke/t01_echo"
        assert kwargs["base_url"].endswith("/v1")
        return driver.HermesOutcome(
            task_id=task_id, passed=True, score=0.9, reason="ok", seconds=12.0
        ), "tail"

    monkeypatch.setattr(driver, "run_task", fake_run_task)
    _, score = get_suite("hermes-bench").run_external(
        Task(task_id="t01_terminal_smoke/t01_echo", prompt=""),
        {
            "model": "m",
            "config": BenchConfig(model="m"),
            "run_id": "r",
            "workdir": tmp_path,
        },
    )
    assert score.passed and score.score == 0.9 and "12s" in score.details
