"""Agent loop tests: fence parsing, turn loop, NL2Repo verify mapping."""

from __future__ import annotations

from benchharness.agent_loop import AgentTranscript, extract_shell, run_shell_loop
from benchharness.sandbox import ExecResult


def test_extract_shell_first_fence_only():
    assert extract_shell("```shell\necho a\n```") == "echo a"
    assert extract_shell("```SHELL\nX\n```") == "X"
    assert extract_shell("no fences here") == ""
    two = "```shell\none\n```\ntext\n```shell\ntwo\n```"
    assert extract_shell(two) == "one"


def _scripted(chats):
    it = iter(chats)

    def chat_fn(messages, cap):
        return next(it), 5, 7

    return chat_fn


def test_loop_runs_commands_and_returns_final():
    seen: list[str] = []

    def exec_fn(cmd):
        seen.append(cmd)
        return ExecResult(0, "ok", "")

    t = run_shell_loop(
        _scripted(["```shell\nls\n```", "```shell\npwd\n```", "all done"]),
        exec_fn, "sys", "do things", max_turns=5,
    )
    assert seen == ["ls", "pwd"]
    assert t.turns == 3 and t.final_text == "all done" and not t.capped
    assert (t.prompt_tokens, t.completion_tokens) == (15, 21)


def test_loop_respects_turn_cap():
    t = run_shell_loop(
        _scripted(["```shell\nx\n```"] * 5),
        lambda cmd: ExecResult(0, "", ""), "sys", "go", max_turns=2,
    )
    assert t.capped and t.turns == 2 and len(t.commands) == 2


def test_loop_survives_exec_errors():
    def boom(cmd):
        raise OSError("nope")

    t = run_shell_loop(_scripted(["```shell\nx\n```", "recovered"]),
                       boom, "sys", "go", max_turns=5)
    assert t.final_text == "recovered" and t.turns == 2


def test_nl2repo_verify_mapping(tmp_path, monkeypatch):
    import benchharness.agent_loop as loop_mod
    from benchharness.registry import get_suite
    from benchharness.suites.base import Task

    class FakeShell:
        def __init__(self, code):
            self.code = code
            self.stopped = False

        def exec(self, cmd, timeout_secs=0):
            return ExecResult(self.code, "out", "err")

        def stop(self):
            self.stopped = True

    def fake_loop(client, model, image, prompt, max_turns=0, max_tokens=0,
                  exec_timeout_secs=0):
        assert image == "img:1"
        t = AgentTranscript(turns=3, commands=["a"], final_text="done")
        return t, FakeShell(0)

    monkeypatch.setattr(loop_mod, "docker_shell_loop", fake_loop)
    adapter = get_suite("nl2repo")
    task = Task(task_id="t", prompt="build it", reference="",
                metadata={"image": "img:1", "verify_cmd": "pytest -q"})
    excerpt, score = adapter.run_external(
        task, {"model": "m", "config": None, "run_id": "r",
               "workdir": tmp_path, "client": object()})
    assert score.passed and "exit=0" in score.details

    def fake_fail(*a, **k):
        return AgentTranscript(turns=1), FakeShell(1)

    monkeypatch.setattr(loop_mod, "docker_shell_loop", fake_fail)
    _, score2 = adapter.run_external(
        task, {"model": "m", "config": None, "run_id": "r",
               "workdir": tmp_path, "client": object()})
    assert not score2.passed

    try:
        adapter.run_external(
            Task(task_id="t", prompt="p", reference="", metadata={}),
            {"model": "m", "config": None, "run_id": "r",
             "workdir": tmp_path, "client": object()})
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "evaluation_image" in str(exc)
