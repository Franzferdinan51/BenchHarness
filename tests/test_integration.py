"""Integration tests over the mock LM transport: client, runner, tool loop."""

from __future__ import annotations

from benchharness.config import BenchConfig
from benchharness.lm_client import LMStudioClient
from benchharness.registry import REGISTRY, get_suite, list_suites
from benchharness.runner import run_suites
from benchharness.schema import read_results
from benchharness.suites.base import Task
from tests.conftest import make_transport


def test_client_ping_and_chat(mock_config):
    client = LMStudioClient(mock_config, transport=make_transport("hello"))
    assert client.ping()["ok"]
    assert client.resolve_model() == "test-llm"
    resp = client.chat([{"role": "user", "content": "hi"}], max_tokens=8)
    assert client.extract_text(resp) == "hello"
    assert client.extract_usage(resp) == (10, 5)
    # embeddings hidden from discovery
    assert [m.id for m in client.list_models()] == ["test-llm"]
    client.close()


def test_client_400_fails_fast_with_body(mock_config):
    import httpx

    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(400, json={"error": {"message": "engine aborted"}})

    client = LMStudioClient(mock_config, transport=httpx.MockTransport(handler))
    try:
        client.chat([{"role": "user", "content": "hi"}], max_tokens=8)
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "engine aborted" in str(exc)
    assert len(calls) == 1  # no retries on 400
    client.close()


def test_client_500_retries_then_succeeds(mock_config):
    import httpx

    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if len(calls) == 1:
            return httpx.Response(500, json={"error": "busy"})
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})

    mock_config.max_retries = 2
    client = LMStudioClient(mock_config, transport=httpx.MockTransport(handler))
    resp = client.chat([{"role": "user", "content": "hi"}], max_tokens=8)
    assert client.extract_text(resp) == "ok"
    assert client.extract_finish_reason(resp) == "stop"
    assert len(calls) == 2
    client.close()


def test_client_auto_model_pick():
    cfg = BenchConfig(model=None)
    client = LMStudioClient(cfg, transport=make_transport("x"))
    assert client.resolve_model() == "test-llm"
    client.close()


def test_registry_covers_goal_suites():
    assert len(REGISTRY) == 20  # 19 goal suites + demo
    for name in ("tb-terminus", "tb-claude", "tb-hermes", "swe-verified", "swe-pro",
                 "swe-multilingual", "deepswe", "frontier-bench", "nl2repo",
                 "swe-atlas-qna", "hle", "hle-tools", "gpqa-diamond",
                 "mcp-atlas", "toolathlon", "widesearch", "browsecomp",
                 "claweval", "hermes-bench", "demo"):
        assert name in REGISTRY, name
    assert len(list_suites("coding")) == 10
    assert len(list_suites("agentic")) == 6


def test_golden_demo_run(tmp_path, mock_config, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # route runner's client through the mock transport
    import benchharness.runner as runner_mod
    real_client = runner_mod.LMStudioClient
    transport = make_transport("PINEAPPLE then 42")

    class MockClient(real_client):
        def __init__(self, config):
            super().__init__(config, transport=transport)

    monkeypatch.setattr(runner_mod, "LMStudioClient", MockClient)
    mock_config.out_dir = tmp_path / "bench-results"
    run_dir, summary = run_suites(["demo"], mock_config)
    assert summary.total == 2 and summary.passed == 2
    assert (run_dir / "summary.json").is_file()
    assert len(read_results(run_dir / "results.jsonl")) == 2


def test_max_tokens_override_reaches_server(mock_config):
    import httpx
    import json as _json
    from benchharness.runner import evaluate_task

    seen: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(_json.loads(request.content.decode())["max_tokens"])
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "PINEAPPLE"},
                         "finish_reason": "stop"}]})

    mock_config.max_tokens_override = 77
    client = LMStudioClient(mock_config, transport=httpx.MockTransport(handler))
    res = evaluate_task(client, get_suite("demo"),
                        Task(task_id="t", prompt="p", reference="PINEAPPLE"),
                        "m", "r", mock_config)
    assert seen == [77] and res.passed
    client.close()


def test_tasks_run_in_parallel(tmp_path, mock_config, monkeypatch):
    """4 x 0.25s mock tasks with jobs=4 must finish well under sequential time."""
    import time
    import httpx
    import benchharness.runner as runner_mod
    real_client = runner_mod.LMStudioClient

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/chat/completions"):
            time.sleep(0.25)
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "PINEAPPLE 42"},
                             "finish_reason": "stop"}]})
        return httpx.Response(200, json={"data": []})

    class MockClient(real_client):
        def __init__(self, config):
            super().__init__(config, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(runner_mod, "LMStudioClient", MockClient)
    mock_config.out_dir = tmp_path / "bench-results"
    mock_config.jobs = 4
    started = time.monotonic()
    _, summary = run_suites(["demo", "demo"], mock_config)
    elapsed = time.monotonic() - started
    assert summary.total == 4
    assert elapsed < 0.9, f"expected parallel (~0.25s), took {elapsed:.2f}s"


def test_per_task_timeout_records_error(tmp_path, mock_config, monkeypatch):
    import time
    import httpx
    import benchharness.runner as runner_mod
    real_client = runner_mod.LMStudioClient

    def handler(request: httpx.Request) -> httpx.Response:
        time.sleep(3)  # simulated hang; timeout must cut it off
        return httpx.Response(200, json={"choices": []})

    class MockClient(real_client):
        def __init__(self, config):
            super().__init__(config, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(runner_mod, "LMStudioClient", MockClient)
    mock_config.out_dir = tmp_path / "bench-results"
    mock_config.per_task_timeout_secs = 0.2
    started = time.monotonic()
    _, summary = run_suites(["demo"], mock_config)
    assert time.monotonic() - started < 10
    assert summary.errors == 2
    assert summary.total == 2


def test_judge_path_overrides_heuristic(mock_config):
    import httpx
    import json as _json
    from benchharness.runner import evaluate_task

    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = _json.loads(request.content.decode())
        text = body["messages"][-1]["content"]
        calls.append(text)
        if "Judge whether" in text:
            content = "reasoning: matches\ncorrect: yes"
        else:
            content = "something that does not contain the answer"
        return httpx.Response(200, json={
            "choices": [{"message": {"content": content},
                         "finish_reason": "stop"}]})

    from benchharness.lm_client import LMStudioClient
    client = LMStudioClient(mock_config, transport=httpx.MockTransport(handler))
    mock_config.judge_enabled = True
    res = evaluate_task(client, get_suite("hle"),
                        Task(task_id="t", prompt="Capital of France?",
                             reference="Paris"),
                        "m", "r", mock_config)
    assert len(calls) == 2  # task call + judge call
    assert res.passed and res.details.startswith("judge=pass")
    assert "heuristic=" in res.details

    # default off: single call, heuristic verdict stands
    mock_config.judge_enabled = False
    calls.clear()
    res2 = evaluate_task(client, get_suite("hle"),
                         Task(task_id="t", prompt="Capital of France?",
                              reference="Paris"),
                         "m", "r", mock_config)
    assert len(calls) == 1 and not res2.passed
    client.close()


def test_tool_loop_executes_python(tmp_path, mock_config, monkeypatch):
    from benchharness.runner import run_tool_loop
    client = LMStudioClient(mock_config,
                            transport=make_transport("```tool\nprint(7*8)\n```"))
    adapter = get_suite("hle-tools")
    task = Task(task_id="t", prompt="What is 7*8?", reference="56",
                metadata={"tool_loop": "run_python", "tool_rounds": 1})
    # First round emits a tool fence; loop appends stdout and re-asks (mock
    # repeats the fence, loop ends after rounds). Just assert it terminates
    # and the round-trip executed without error.
    out, pt, ct = run_tool_loop(client, adapter, task, "test-llm", 64)
    assert isinstance(out, str) and pt > 0
    client.close()


def test_soft_requirement_falls_back_to_smoke(tmp_path, mock_config, monkeypatch):
    """Missing `datasets` must NOT skip the suite; smoke samples run instead."""
    import benchharness.runner as runner_mod
    import benchharness.suites.reasoning as reasoning_mod
    real_client = runner_mod.LMStudioClient
    transport = make_transport("Answer: B")

    class MockClient(real_client):
        def __init__(self, config):
            super().__init__(config, transport=transport)

    monkeypatch.setattr(runner_mod, "LMStudioClient", MockClient)
    monkeypatch.setattr(reasoning_mod, "_load_hf_dataset", lambda *a, **k: None)
    mock_config.out_dir = tmp_path / "bench-results"
    run_dir, summary = run_suites(["gpqa-diamond"], mock_config)
    assert summary.total == 2 and summary.skipped == 0
    assert summary.passed == 1  # smoke-1 wants B, smoke-2 wants C


def test_hard_requirement_skips_suite(tmp_path, mock_config, monkeypatch):
    import benchharness.runner as runner_mod
    real_client = runner_mod.LMStudioClient

    class MockClient(real_client):
        def __init__(self, config):
            super().__init__(config, transport=make_transport("x"))

    monkeypatch.setattr(runner_mod, "LMStudioClient", MockClient)
    monkeypatch.setattr("shutil.which", lambda name: None)  # no docker
    mock_config.out_dir = tmp_path / "bench-results"
    _, summary = run_suites(["tb-terminus"], mock_config)
    assert summary.skipped == 1 and summary.total == 1


def test_swe_patch_scoring():
    adapter = get_suite("swe-verified")
    ref = "diff --git a/x.py b/x.py\n- a\n+ b"
    task = Task(task_id="t", prompt="p", reference=ref)
    exact = adapter.score(f"```diff\n{ref}\n```", task)
    assert exact.passed and exact.score == 1.0
    empty = adapter.score("I have no idea", task)
    assert not empty.passed and empty.score == 0.0


def test_swe_docker_grade_preferred_and_fallback(tmp_path, monkeypatch):
    import benchharness.swe_eval as swe_eval
    from benchharness.registry import get_suite
    from benchharness.suites.base import Task

    adapter = get_suite("swe-verified")
    adapter.prepare(tmp_path)
    ref = "diff --git a/x.py b/x.py\n- a\n+ b"
    task = Task(task_id="t", prompt="p", reference=ref)
    out = f"```diff\n{ref}\n```"

    monkeypatch.setattr(swe_eval, "docker_grading_available", lambda: True)
    monkeypatch.setattr(swe_eval, "grade_with_docker",
                        lambda *a, **k: swe_eval.DockerGrade(True, "official"))
    assert adapter.score(out, task).details == "official"

    monkeypatch.setattr(swe_eval, "grade_with_docker",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    fallen = adapter.score(out, task)
    assert fallen.passed and fallen.details.startswith("docker eval crashed")

    monkeypatch.setattr(swe_eval, "docker_grading_available", lambda: False)
    assert "official" not in adapter.score(out, task).details


def test_ensure_image_skips_present_and_forces_amd64(monkeypatch):
    import benchharness.swe_eval as swe_eval

    calls: list[list[str]] = []

    class FakeProc:
        def __init__(self, stdout="", returncode=0, stderr=""):
            self.stdout = stdout
            self.returncode = returncode
            self.stderr = stderr

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:2] == ["docker", "images"]:
            return FakeProc(stdout="" if "missing" in cmd[-1] else "abc123\n")
        return FakeProc(stdout="pulled")

    monkeypatch.setattr(swe_eval.subprocess, "run", fake_run)
    monkeypatch.setattr(swe_eval.platform, "machine", lambda: "arm64")
    swe_eval.ensure_image("img:present")
    assert calls == [["docker", "images", "-q", "img:present"]]
    swe_eval.ensure_image("img:missing")
    assert calls[-1] == ["docker", "pull", "--platform", "linux/amd64", "img:missing"]


def test_eval_dataset_mapping():
    import benchharness.swe_eval as swe_eval
    assert swe_eval.eval_dataset_for("princeton-nlp/SWE-bench_Verified") == \
        "SWE-bench/SWE-bench_Verified"
    assert swe_eval.eval_dataset_for("other/ds") == "other/ds"


def test_swe_docker_env_kill_switch(monkeypatch):
    import benchharness.swe_eval as swe_eval
    monkeypatch.setenv("BENCH_SWE_DOCKER", "0")
    assert swe_eval.docker_grading_available() is False


def test_swe_pro_stays_heuristic(tmp_path):
    from benchharness.registry import get_suite
    from benchharness.suites.base import Task
    adapter = get_suite("swe-pro")
    adapter.prepare(tmp_path)
    ref = "diff --git a/x.py b/x.py\n- a\n+ b"
    assert "ScaleAI" in adapter.source
    assert adapter._docker_grade(Task(task_id="t", prompt="p", reference=ref),
                                 ref) is None
