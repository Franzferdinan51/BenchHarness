"""Harbor driver tests: command building, result parsing, TB mapping."""

from __future__ import annotations

import json

from benchharness.harbor_driver import (
    build_run_command,
    list_tasks,
    parse_job_dir,
    parse_trial_result,
)


def test_build_run_command_terminus_routing(tmp_path):
    cmd = build_run_command(
        "terminal-bench@2.0", "terminus-2", "openai/test-model",
        tmp_path, "job1", include_task="chess-best-move", n_tasks=1,
        agent_kwargs={"api_base": "http://127.0.0.1:1234/v1"},
        agent_env={"OPENAI_API_KEY": "lm-studio"},
    )
    assert cmd[:6] == ["harbor", "run", "-d", "terminal-bench@2.0", "-a", "terminus-2"]
    assert "--ak" in cmd and "api_base=http://127.0.0.1:1234/v1" in cmd
    assert "--ae" in cmd and "OPENAI_API_KEY=lm-studio" in cmd
    assert "-i" in cmd and "chess-best-move" in cmd
    assert "-l" in cmd


def _trial(tmp_path, name, **fields):
    d = tmp_path / name
    d.mkdir()
    base = {"task_name": "chess-best-move", "trial_name": name,
            "verifier_result": {"rewards": {"reward": 1.0}},
            "exception_info": None,
            "agent_execution": {"duration": 12.5}}
    base.update(fields)
    (d / "result.json").write_text(json.dumps(base))
    return d


def test_parse_trial_pass_fail_error(tmp_path):
    ok = _trial(tmp_path, "t1")
    assert parse_trial_result(ok / "result.json").passed
    fail = _trial(tmp_path, "t2", verifier_result={"rewards": {"reward": 0.0}})
    assert not parse_trial_result(fail / "result.json").passed
    err = _trial(tmp_path, "t3", exception_info={"exception_type": "Boom",
                                                 "message": "kaput"})
    parsed = parse_trial_result(err / "result.json")
    assert not parsed.passed and "Boom" in parsed.error
    assert (_trial(tmp_path, "t4", verifier_result=None) and
            not parse_trial_result(tmp_path / "t4" / "result.json").passed)
    assert parse_trial_result(tmp_path / "missing.json") is None
    (tmp_path / "junk.json").write_text("not json")
    assert parse_trial_result(tmp_path / "junk.json") is None


def test_parse_job_dir_collects_trials(tmp_path):
    _trial(tmp_path, "trial-a")
    _trial(tmp_path, "trial-b", task_name="other",
           verifier_result={"rewards": {"reward": 0.0}})
    outcomes = parse_job_dir(tmp_path)
    assert len(outcomes) == 2
    assert outcomes[0].task_name == "chess-best-move"
    assert parse_job_dir(tmp_path / "nope") == []


def test_list_tasks_from_task_toml(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "task.toml").write_text("")
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "task.toml").write_text("")
    (tmp_path / "notes.txt").write_text("")
    assert list_tasks(tmp_path) == ["a", "b"]


def test_tb_run_external_maps_outcome(tmp_path, monkeypatch):
    from benchharness.config import BenchConfig
    from benchharness.registry import get_suite
    from benchharness.suites.base import Task

    seen: dict = {}

    def fake_run_job(cmd, timeout, env=None):
        seen["cmd"] = cmd
        return 0, "job tail"

    def fake_parse(job_dir):
        from benchharness.harbor_driver import TrialOutcome
        return [TrialOutcome(task_name="t", passed=True, score=1.0,
                             rewards={"reward": 1}, seconds=3.0)]

    # run_external imports inside the function; patch at driver module level
    import benchharness.harbor_driver as driver
    monkeypatch.setattr(driver, "run_job", fake_run_job)
    monkeypatch.setattr(driver, "parse_job_dir", fake_parse)

    adapter = get_suite("tb-terminus")
    cfg = BenchConfig(model="m", api_key="k")
    excerpt, score = adapter.run_external(
        Task(task_id="chess-best-move", prompt="", reference="",
             metadata={"harbor_task": "chess-best-move"}),
        {"model": "m", "config": cfg, "run_id": "r", "workdir": tmp_path})
    assert score.passed and excerpt == "job tail"
    joined = " ".join(seen["cmd"])
    assert "--agent" not in joined  # short flags used
    assert "-a terminus-2" in joined and "api_base=" in joined


def test_tb_hermes_requires_cli_and_skips_api_base(tmp_path, monkeypatch):
    from benchharness.config import BenchConfig
    from benchharness.registry import get_suite
    from benchharness.suites.base import Task

    adapter = get_suite("tb-hermes")
    kinds = [(r.kind, r.name) for r in adapter.requirements()]
    assert ("cli", "hermes") in kinds

    seen: dict = {}

    def fake_run_job(cmd, timeout, env=None):
        seen["cmd"] = cmd
        return 0, "tail"

    def fake_parse(job_dir):
        from benchharness.harbor_driver import TrialOutcome
        return [TrialOutcome(task_name="t", passed=True, score=1.0)]

    import benchharness.harbor_driver as driver
    monkeypatch.setattr(driver, "run_job", fake_run_job)
    monkeypatch.setattr(driver, "parse_job_dir", fake_parse)
    _, score = adapter.run_external(
        Task(task_id="t", prompt="", reference="", metadata={"harbor_task": "t"}),
        {"model": "m", "config": BenchConfig(model="m"), "run_id": "r",
         "workdir": tmp_path})
    assert score.passed
    joined = " ".join(seen["cmd"])
    assert "-a hermes" in joined and "api_base" not in joined


def test_hermes_bench_defers_cleanly():
    from benchharness.registry import get_suite
    tasks = get_suite("hermes-bench").tasks()
    assert len(tasks) == 1 and "skip_reason" in tasks[0].metadata
