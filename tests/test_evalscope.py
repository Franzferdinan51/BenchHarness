"""EvalScope driver tests: sample parsing (shapes from the stub probe),
run_one plumbing (subprocess stubbed), adapter wiring (run_one stubbed)."""

import json

from benchharness.evalscope_driver import (
    EvalScopeOutcome,
    parse_samples,
    run_one,
)


def _claw_row(passed, task_score=1.0, err=""):
    return {"value": {"task_score": task_score, "passed": passed,
                      "error_rate": 0.0 if passed else 1.0,
                      "judge_score": task_score},
            "status": "ok",
            "metadata": {"task_id": "T1", "error": err,
                         "trace_path": "/tmp/tr.jsonl"}}


def test_parse_claw_pass_and_fail():
    oc = parse_samples("T1", [_claw_row(1.0)])
    assert oc.passed and oc.score == 1.0
    assert oc.trace_path == "/tmp/tr.jsonl"
    oc = parse_samples("T1", [_claw_row(0.0, 0.0, "boom")])
    assert not oc.passed and oc.score == 0.0 and "boom" in oc.details


def test_parse_pass_cubed_requires_all_trials():
    oc = parse_samples("T1", [_claw_row(1.0), _claw_row(1.0), _claw_row(0.0, 0.0)])
    assert not oc.passed  # 2/3 is not Pass^3
    assert "trials=3" in oc.details
    oc = parse_samples("T1", [_claw_row(1.0)] * 3)
    assert oc.passed


def test_parse_deep_swe_acc():
    rows = [{"value": {"acc": 1.0}, "status": "ok",
             "metadata": {"reward": 1.0, "pier_job_result_path": "/tmp/p.json"}}]
    oc = parse_samples("t1", rows)
    assert oc.passed and oc.score == 1.0
    assert oc.trace_path == "/tmp/p.json"
    oc = parse_samples("t1", [{"value": {"acc": 0.0}, "status": "ok",
                               "metadata": {}}])
    assert not oc.passed


def test_parse_empty_samples():
    oc = parse_samples("t1", [])
    assert not oc.passed and oc.error == "parse"


def test_run_one_parses_samples_line(tmp_path, monkeypatch):
    import subprocess

    import benchharness.evalscope_driver as driver

    samples = [_claw_row(1.0)]
    stdout = "noise\nBENCH_SAMPLES_JSON:" + json.dumps(samples) + "\n"

    class FakeProc:
        returncode = 0
        stderr = ""

    FakeProc.stdout = stdout

    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr(subprocess, "run", fake_run)
    oc = run_one("claw_eval", "T1", model="m", api_base="http://x/v1",
                 api_key="k", split="general", trials=1, work_dir=tmp_path)
    assert isinstance(oc, EvalScopeOutcome)
    assert oc.passed and oc.score == 1.0
    assert "claw_eval" in seen["cmd"] and "T1" in seen["cmd"]


def test_run_one_no_samples_reports_tail(tmp_path, monkeypatch):
    import subprocess

    import benchharness.evalscope_driver as driver  # noqa: F401

    class FakeProc:
        returncode = 1
        stdout = "nothing here"
        stderr = "kaboom"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeProc())
    oc = run_one("deep_swe", "t1", model="m", api_base="u", api_key="k",
                 work_dir=tmp_path)
    assert not oc.passed and oc.error == "no-result" and "kaboom" in oc.details


def test_run_one_timeout(tmp_path, monkeypatch):
    import subprocess

    import benchharness.evalscope_driver as driver  # noqa: F401

    def fake_run(*a, **k):
        raise subprocess.TimeoutExpired(cmd=[], timeout=1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    oc = run_one("claw_eval", "T1", model="m", api_base="u", api_key="k",
                 work_dir=tmp_path, timeout_secs=1)
    assert not oc.passed and oc.error == "timeout"


def test_claweval_run_external_uses_driver(tmp_path, monkeypatch):
    import benchharness.evalscope_driver as driver
    from benchharness.config import BenchConfig
    from benchharness.registry import get_suite
    from benchharness.suites.base import Task

    def fake_run_one(benchmark, task_id, **kwargs):
        assert benchmark == "claw_eval"
        assert kwargs["split"] == "multi_turn"
        assert kwargs["trials"] == 3
        return EvalScopeOutcome(task_id, 1.0, True, "score=1.000 trials=3")

    monkeypatch.setattr(driver, "run_one", fake_run_one)
    monkeypatch.setenv("BENCH_CLAW_TRIALS", "3")
    adapter = get_suite("claweval")
    task = Task(task_id="T9", prompt="q", metadata={"split": "multi_turn"})
    out, score = adapter.run_external(
        task, {"model": "m", "config": BenchConfig(model="m"),
               "run_id": "r", "workdir": tmp_path})
    assert score.passed and "trials=3" in out


def test_deepswe_tasks_from_snapshot_dir(tmp_path, monkeypatch):
    import benchharness.suites.coding as coding_mod
    from benchharness.registry import get_suite

    tasks_dir = tmp_path / "tasks"
    good = tasks_dir / "t1"
    good.mkdir(parents=True)
    (good / "task.toml").write_text("[task]\n")
    (good / "instruction.md").write_text("Do the thing.")
    (tasks_dir / "notask").mkdir()
    monkeypatch.setattr(coding_mod, "_load_deepswe_tasks_dir",
                        lambda: tasks_dir)
    tasks = get_suite("deepswe").tasks()
    assert [t.task_id for t in tasks] == ["t1"]
    assert tasks[0].prompt == "Do the thing."


def test_deepswe_run_external_uses_driver(tmp_path, monkeypatch):
    import benchharness.evalscope_driver as driver
    from benchharness.config import BenchConfig
    from benchharness.registry import get_suite
    from benchharness.suites.base import Task

    def fake_run_one(benchmark, task_id, **kwargs):
        assert benchmark == "deep_swe"
        return EvalScopeOutcome(task_id, 0.0, False, "score=0.000 trials=1",
                                error="")

    monkeypatch.setattr(driver, "run_one", fake_run_one)
    adapter = get_suite("deepswe")
    task = Task(task_id="t1", prompt="p")
    _, score = adapter.run_external(
        task, {"model": "m", "config": BenchConfig(model="m"),
               "run_id": "r", "workdir": tmp_path})
    assert not score.passed and score.score == 0.0


def test_list_bundled_tasks_parses_ids(monkeypatch):
    import subprocess

    import benchharness.evalscope_driver as driver

    class FakeProc:
        returncode = 0
        stdout = 'BENCH_TASKS_JSON:["a", "b"]\n'
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeProc())
    assert driver.list_bundled_tasks("toolathlon") == ["a", "b"]


def test_list_bundled_tasks_empty_on_failure(monkeypatch):
    import subprocess

    import benchharness.evalscope_driver as driver

    def boom(*a, **k):
        raise OSError("nope")

    monkeypatch.setattr(subprocess, "run", boom)
    assert driver.list_bundled_tasks("toolathlon") == []


def test_run_one_forwards_extra_params(tmp_path, monkeypatch):
    import subprocess

    import benchharness.evalscope_driver as driver

    class FakeProc:
        returncode = 0
        stdout = 'BENCH_SAMPLES_JSON:[{"value": {"acc": 1.0}, "metadata": {}}]\n'
        stderr = ""

    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr(subprocess, "run", fake_run)
    oc = driver.run_one("toolathlon", "ab-testing", model="m", api_base="u",
                        api_key="k", work_dir=tmp_path,
                        extra_params={"task_list": ["ab-testing"]})
    assert oc.passed
    assert json.loads(seen["cmd"][-1]) == {"task_list": ["ab-testing"]}


def test_toolathlon_tasks_and_run_external(tmp_path, monkeypatch):
    import benchharness.evalscope_driver as driver
    from benchharness.config import BenchConfig
    from benchharness.registry import get_suite
    from benchharness.suites.base import Task

    monkeypatch.setattr(driver, "list_bundled_tasks", lambda b: ["t1", "t2"])
    tasks = get_suite("toolathlon").tasks()
    assert [t.task_id for t in tasks] == ["t1", "t2"]

    def fake_run_one(benchmark, task_id, **kwargs):
        assert benchmark == "toolathlon"
        assert kwargs["extra_params"] == {"task_list": ["t1"]}
        return EvalScopeOutcome(task_id, 1.0, True, "score=1.000 trials=1")

    monkeypatch.setattr(driver, "run_one", fake_run_one)
    adapter = get_suite("toolathlon")
    assert adapter.status == "wired"
    _, score = adapter.run_external(
        Task(task_id="t1", prompt="p"),
        {"model": "m", "config": BenchConfig(model="m"),
         "run_id": "r", "workdir": tmp_path})
    assert score.passed


def test_evalscope_requirement_kind(tmp_path, monkeypatch):
    from benchharness.suites.base import Requirement, _requirement_missing

    monkeypatch.setenv("BENCH_EVALSCOPE_PYTHON", str(tmp_path / "nope"))
    assert _requirement_missing(Requirement("evalscope", "claw_eval", ""))
    py = tmp_path / "py"
    py.write_text("")
    monkeypatch.setenv("BENCH_EVALSCOPE_PYTHON", str(py))
    assert not _requirement_missing(Requirement("evalscope", "claw_eval", ""))
