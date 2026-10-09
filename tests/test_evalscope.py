"""EvalScope driver tests: sample parsing (shapes from the stub probe),
run_one plumbing (subprocess stubbed), adapter wiring (run_one stubbed)."""

import json

from benchharness.evalscope_driver import (
    EvalScopeOutcome,
    parse_samples,
    run_one,
)


def _claw_row(passed, task_score=1.0, err=""):
    return {
        "value": {
            "task_score": task_score,
            "passed": passed,
            "error_rate": 0.0 if passed else 1.0,
            "judge_score": task_score,
        },
        "status": "ok",
        # fixture strings, never created on disk
        "metadata": {
            "task_id": "T1",
            "error": err,
            "trace_path": "/tmp/tr.jsonl",  # noqa: S108
        },
    }


def test_parse_claw_pass_and_fail():
    oc = parse_samples("T1", [_claw_row(1.0)])
    assert oc.passed and oc.score == 1.0
    assert oc.trace_path == "/tmp/tr.jsonl"  # noqa: S108
    oc = parse_samples("T1", [_claw_row(0.0, 0.0, "boom")])
    assert not oc.passed and oc.score == 0.0 and "boom" in oc.details


def test_parse_pass_cubed_requires_all_trials():
    oc = parse_samples("T1", [_claw_row(1.0), _claw_row(1.0), _claw_row(0.0, 0.0)])
    assert not oc.passed  # 2/3 is not Pass^3
    assert "trials=3" in oc.details
    oc = parse_samples("T1", [_claw_row(1.0)] * 3)
    assert oc.passed


def test_row_outcome_mcp_pass_shape():
    from benchharness.evalscope_driver import row_outcome

    assert row_outcome({"coverage_score": 0.8, "pass": 1.0}) == (0.8, True)
    assert row_outcome({"coverage_score": 0.2, "pass": 0.0}) == (0.2, False)
    # parse_samples uses the same helper
    oc = parse_samples(
        "t", [{"value": {"coverage_score": 1.0, "pass": 1.0}, "metadata": {}}]
    )
    assert oc.passed and oc.score == 1.0


def test_run_batch_parses_rows_and_limit(tmp_path, monkeypatch):
    import subprocess

    import benchharness.evalscope_driver as driver

    rows = [
        {
            "value": {"coverage_score": 0.5, "pass": 0.0},
            "status": "success",
            "metadata": {},
            "prompt": "do x",
        }
    ]

    class FakeProc:
        returncode = 0
        stdout = ""
        stderr = ""

    FakeProc.stdout = "BENCH_SAMPLES_JSON:" + json.dumps(rows) + "\n"
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr(subprocess, "run", fake_run)
    got, error = driver.run_batch(
        "mcp_atlas", model="m", api_base="u", api_key="k", limit=7, work_dir=tmp_path
    )
    assert error == "" and got == rows
    assert seen["cmd"][-1] == "7" and seen["cmd"][-2] == "{}"


def test_run_batch_error_paths(tmp_path, monkeypatch):
    import subprocess

    import benchharness.evalscope_driver as driver

    class FakeProc:
        returncode = 2
        stdout = "nope"
        stderr = "bad"

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: FakeProc())
    rows, error = driver.run_batch(
        "mcp_atlas", model="m", api_base="u", api_key="k", limit=1, work_dir=tmp_path
    )
    assert rows == [] and "rc=2" in error

    def boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd=[], timeout=1)

    monkeypatch.setattr(subprocess, "run", boom)
    rows, error = driver.run_batch(
        "mcp_atlas", model="m", api_base="u", api_key="k", limit=1, work_dir=tmp_path
    )
    assert rows == [] and "timeout" in error


def test_mcp_recall_fallback_without_service(tmp_path, monkeypatch):
    import benchharness.suites.agentic as agentic_mod
    from benchharness.config import BenchConfig
    from benchharness.suites.base import Task

    monkeypatch.setattr(agentic_mod, "_mcp_env_reachable", lambda *a, **k: False)
    adapter = agentic_mod.McpAtlasAdapter()
    task = Task(
        task_id="t",
        prompt="p",
        reference="claims",
        metadata={"raw_prompt": "p", "ds_index": 0},
    )
    assert (
        adapter.run_external(
            task,
            {
                "model": "m",
                "config": BenchConfig(model="m"),
                "run_id": "r",
                "workdir": tmp_path,
            },
        )
        is None
    )
    out = adapter.score("claims words here", task)
    assert "recall mode" in out.details


def test_mcp_agent_mode_cache_match_and_exclude(tmp_path, monkeypatch):
    import benchharness.evalscope_driver as driver
    import benchharness.suites.agentic as agentic_mod
    from benchharness.config import BenchConfig
    from benchharness.suites.base import Task

    monkeypatch.setattr(agentic_mod.McpAtlasAdapter, "_agent_mode", lambda self: True)
    calls = []

    def fake_batch(benchmark, **kwargs):
        calls.append(kwargs["limit"])
        assert kwargs["extra_params"]["mcp_server_url"].endswith(":1984")
        return (
            [
                {
                    "value": {"coverage_score": 1.0, "pass": 1.0},
                    "status": "success",
                    "metadata": {},
                    "prompt": "p0",
                }
            ]
            if kwargs["limit"] == 1
            else [
                {
                    "value": {"coverage_score": 1.0, "pass": 1.0},
                    "status": "success",
                    "metadata": {},
                    "prompt": "p0",
                },
                {
                    "value": {"coverage_score": 0.0, "pass": 0.0},
                    "status": "success",
                    "metadata": {},
                    "prompt": "p1",
                },
            ]
        ), ""

    monkeypatch.setattr(driver, "run_batch", fake_batch)
    adapter = agentic_mod.McpAtlasAdapter()
    ctx = {
        "model": "m",
        "config": BenchConfig(model="m"),
        "run_id": "r",
        "workdir": tmp_path,
    }

    def task(i):
        return Task(
            task_id=f"t{i}",
            prompt=f"p{i} + tools",
            metadata={"raw_prompt": f"p{i}", "ds_index": i},
        )

    _, s0 = adapter.run_external(task(0), ctx)
    assert s0.passed and s0.score == 1.0 and "agent mode" in s0.details
    # second call reuses cache (no new batch while covered)
    _, s0b = adapter.run_external(task(0), ctx)
    assert s0b.passed and calls == [1]
    # excluded task (prompt missing from rows)
    _, sx = adapter.run_external(
        Task(task_id="tx", prompt="px", metadata={"raw_prompt": "px", "ds_index": 0}),
        ctx,
    )
    assert not sx.passed and "excluded" in sx.details
    # growth: index beyond coverage reruns with bigger limit
    _, s1 = adapter.run_external(task(1), ctx)
    assert not s1.passed and calls == [1, 2]


def test_parse_deep_swe_acc():
    rows = [
        {
            "value": {"acc": 1.0},
            "status": "ok",
            # fixture strings, never created on disk
            "metadata": {
                "reward": 1.0,
                "pier_job_result_path": "/tmp/p.json",  # noqa: S108
            },
        }
    ]
    oc = parse_samples("t1", rows)
    assert oc.passed and oc.score == 1.0
    assert oc.trace_path == "/tmp/p.json"  # noqa: S108
    oc = parse_samples("t1", [{"value": {"acc": 0.0}, "status": "ok", "metadata": {}}])
    assert not oc.passed


def test_parse_empty_samples():
    oc = parse_samples("t1", [])
    assert not oc.passed and oc.error == "parse"


def test_run_one_parses_samples_line(tmp_path, monkeypatch):
    import subprocess

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
    oc = run_one(
        "claw_eval",
        "T1",
        model="m",
        api_base="http://x/v1",
        api_key="k",
        split="general",
        trials=1,
        work_dir=tmp_path,
    )
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
    oc = run_one(
        "deep_swe", "t1", model="m", api_base="u", api_key="k", work_dir=tmp_path
    )
    assert not oc.passed and oc.error == "no-result" and "kaboom" in oc.details


def test_run_one_timeout(tmp_path, monkeypatch):
    import subprocess

    import benchharness.evalscope_driver as driver  # noqa: F401

    def fake_run(*a, **k):
        raise subprocess.TimeoutExpired(cmd=[], timeout=1)

    monkeypatch.setattr(subprocess, "run", fake_run)
    oc = run_one(
        "claw_eval",
        "T1",
        model="m",
        api_base="u",
        api_key="k",
        work_dir=tmp_path,
        timeout_secs=1,
    )
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
        task,
        {
            "model": "m",
            "config": BenchConfig(model="m"),
            "run_id": "r",
            "workdir": tmp_path,
        },
    )
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
    monkeypatch.setattr(coding_mod, "_load_deepswe_tasks_dir", lambda: tasks_dir)
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
        return EvalScopeOutcome(task_id, 0.0, False, "score=0.000 trials=1", error="")

    monkeypatch.setattr(driver, "run_one", fake_run_one)
    adapter = get_suite("deepswe")
    task = Task(task_id="t1", prompt="p")
    _, score = adapter.run_external(
        task,
        {
            "model": "m",
            "config": BenchConfig(model="m"),
            "run_id": "r",
            "workdir": tmp_path,
        },
    )
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
    oc = driver.run_one(
        "toolathlon",
        "ab-testing",
        model="m",
        api_base="u",
        api_key="k",
        work_dir=tmp_path,
        extra_params={"task_list": ["ab-testing"]},
    )
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
        {
            "model": "m",
            "config": BenchConfig(model="m"),
            "run_id": "r",
            "workdir": tmp_path,
        },
    )
    assert score.passed


def test_container_base_url_rewrites_localhost(monkeypatch):
    from benchharness.evalscope_driver import container_base_url

    assert (
        container_base_url("http://127.0.0.1:1234/v1")
        == "http://host.docker.internal:1234/v1"
    )
    assert (
        container_base_url("http://localhost:8080/x")
        == "http://host.docker.internal:8080/x"
    )
    assert (
        container_base_url("https://api.example.com/v1") == "https://api.example.com/v1"
    )
    monkeypatch.setenv("BENCH_CONTAINER_HOST", "10.0.0.5")
    assert container_base_url("http://127.0.0.1:1234/v1") == "http://10.0.0.5:1234/v1"


def test_run_one_deep_swe_prefix_and_env(tmp_path, monkeypatch):
    import subprocess

    import benchharness.evalscope_driver as driver

    class FakeProc:
        returncode = 0
        stdout = ""
        stderr = ""

    FakeProc.stdout = 'BENCH_SAMPLES_JSON:[{"value": {"acc": 1.0}}]\n'
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["env"] = kwargs.get("env")
        return FakeProc()

    monkeypatch.setattr(subprocess, "run", fake_run)
    oc = driver.run_one(
        "deep_swe",
        "t1",
        model="mymodel",
        api_base="http://127.0.0.1:1234/v1",
        api_key="k",
        work_dir=tmp_path,
    )
    assert oc.passed
    argv_model = seen["cmd"][seen["cmd"].index("deep_swe") + 3]
    assert argv_model == "openai/mymodel"
    assert seen["env"]["OPENAI_API_BASE"] == "http://host.docker.internal:1234/v1"
    assert seen["env"]["OPENAI_API_KEY"] == "k"

    # other benchmarks: no prefix, no env override
    oc = driver.run_one(
        "claw_eval",
        "T1",
        model="mymodel",
        api_base="http://127.0.0.1:1234/v1",
        api_key="k",
        work_dir=tmp_path,
    )
    assert seen["env"] is None


def test_driver_wraps_deepswe_snapshot_download():
    # The egress proxy only allows ports 80/443, so the driver must
    # re-apply allow_internet after every (self-restoring) download.
    # BENCH_DEEPSWE_ALLOW_INTERNET=0 restores the locked protocol.
    from benchharness.evalscope_driver import DRIVER_SCRIPT

    assert "BENCH_DEEPSWE_ALLOW_INTERNET" in DRIVER_SCRIPT
    assert "_patched_download" in DRIVER_SCRIPT
    assert "allow_internet = true" in DRIVER_SCRIPT


def test_warn_if_outside_home(tmp_path, capsys):
    from pathlib import Path

    from benchharness.evalscope_driver import warn_if_outside_home

    warn_if_outside_home(Path.home())
    assert "warning" not in capsys.readouterr().err
    # /tmp is outside $HOME on macOS (/private/tmp); missing ok too
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        from pathlib import Path

        try:
            Path(td).resolve().relative_to(Path.home().resolve())
            inside = True
        except ValueError:
            inside = False
        warn_if_outside_home(Path(td))
        err = capsys.readouterr().err
        assert ("warning" in err) != inside


def test_evalscope_requirement_kind(tmp_path, monkeypatch):
    from benchharness.suites.base import Requirement, _requirement_missing

    monkeypatch.setenv("BENCH_EVALSCOPE_PYTHON", str(tmp_path / "nope"))
    assert _requirement_missing(Requirement("evalscope", "claw_eval", ""))
    py = tmp_path / "py"
    py.write_text("")
    monkeypatch.setenv("BENCH_EVALSCOPE_PYTHON", str(py))
    assert not _requirement_missing(Requirement("evalscope", "claw_eval", ""))


def test_run_one_kwarg_contract(tmp_path, monkeypatch):
    """Adapters must call run_one with its real kwargs (work_dir, not workdir).

    Regression: all three EvalScope callers passed workdir=, which raised
    TypeError on every live trial while mocks stayed green.
    """
    import inspect

    import benchharness.evalscope_driver as driver
    from benchharness.config import BenchConfig
    from benchharness.suites.base import Task

    sig = inspect.signature(driver.run_one)
    assert "work_dir" in sig.parameters

    seen = {}

    def strict_run_one(benchmark, task_id, **kwargs):
        sig.bind(benchmark, task_id, **kwargs)  # TypeError on bad kwargs
        seen[benchmark] = kwargs
        return driver.EvalScopeOutcome(
            task_id=task_id, score=1.0, passed=True, details="stub"
        )

    import benchharness.suites.agentic as agentic_mod
    import benchharness.suites.coding as coding_mod

    # Adapters import run_one inside run_external, so patch the driver attr.
    monkeypatch.setattr(driver, "run_one", strict_run_one)
    ctx = {
        "model": "m",
        "config": BenchConfig(model="m"),
        "run_id": "r",
        "workdir": tmp_path,
    }
    coding_mod.DeepSweAdapter().run_external(
        Task(task_id="t", prompt="p", metadata={"tasks_dir": str(tmp_path)}), ctx
    )
    agentic_mod.ToolathlonAdapter().run_external(Task(task_id="t", prompt="p"), ctx)
    agentic_mod.ClawEvalAdapter().run_external(
        Task(task_id="t", prompt="p", metadata={"split": "general"}), ctx
    )
    assert set(seen) == {"deep_swe", "toolathlon", "claw_eval"}
    assert all("work_dir" in kw for kw in seen.values())


def test_ensure_claw_agent_image_present(monkeypatch):
    import benchharness.evalscope_driver as driver
    import benchharness.sandbox as sandbox_mod
    from benchharness.sandbox import ExecResult

    # run_local is imported inside the function from sandbox.
    monkeypatch.setattr(
        sandbox_mod, "run_local", lambda *a, **k: ExecResult(0, "img", "")
    )
    assert driver.ensure_claw_agent_image() == "present"


def test_ensure_claw_agent_image_builds_with_docker_io(tmp_path, monkeypatch):
    import benchharness.evalscope_driver as driver
    import benchharness.sandbox as sandbox_mod
    from benchharness.sandbox import ExecResult

    repo = tmp_path / "extracted" / "repo123"
    repo.mkdir(parents=True)
    (repo / "Dockerfile.agent").write_text("FROM x")
    monkeypatch.setattr(
        driver, "claw_official_extract_root", lambda: tmp_path / "extracted"
    )
    seen = []

    def fake_run(argv, **kwargs):
        seen.append(argv)
        if argv[:3] == ["docker", "image", "inspect"]:
            return ExecResult(1, "", "nope")
        return ExecResult(0, "built", "")

    monkeypatch.setattr(sandbox_mod, "run_local", fake_run)
    assert driver.ensure_claw_agent_image() == "built"
    build = next(a for a in seen if len(a) > 1 and a[1] == "build")
    assert "REGISTRY=docker.io" in build
    assert "claw-eval-agent:latest" in build


def test_ensure_claw_agent_image_missing_extract_warns(tmp_path, monkeypatch, capsys):
    import benchharness.evalscope_driver as driver
    import benchharness.sandbox as sandbox_mod
    from benchharness.sandbox import ExecResult

    monkeypatch.setattr(
        driver, "claw_official_extract_root", lambda: tmp_path / "empty"
    )
    monkeypatch.setattr(
        sandbox_mod, "run_local", lambda *a, **k: ExecResult(1, "", "nope")
    )
    assert driver.ensure_claw_agent_image() == "missing-extract"
    assert "DaoCloud" in capsys.readouterr().err
