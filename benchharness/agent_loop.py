"""Generic docker shell-agent loop for build-and-verify suites (NL2Repo first).

Protocol: the model replies with fenced shell blocks to act, or plain text
to finish::

    ```shell
    ls /w && python -m pytest -q
    ```

Only the FIRST fenced block per turn executes (predictable transcripts);
the loop stops at the turn cap or when a reply contains no fence. Tool
results feed back as user messages. Consumers (NL2Repo, later Toolathlon)
run their own verifier command afterwards and grade on its outcome.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

from benchharness.sandbox import ExecResult, run_local

SHELL_FENCE = re.compile(r"```shell\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_shell(text: str) -> str:
    """First ```shell fence body, or '' when the model is done acting."""
    m = SHELL_FENCE.search(text)
    return m.group(1).strip() if m else ""


@dataclass
class AgentTranscript:
    turns: int = 0
    commands: list[str] = field(default_factory=list)
    final_text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    capped: bool = False


def run_shell_loop(
    chat_fn,
    exec_fn,
    system: str,
    user_prompt: str,
    max_turns: int = 20,
    max_tokens: int = 4096,
) -> AgentTranscript:
    """Turn loop over injected chat/exec callables (docker-free, testable).

    chat_fn(messages, max_tokens) -> (text, prompt_tok, completion_tok)
    exec_fn(command) -> ExecResult
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]
    transcript = AgentTranscript()
    text = ""
    for turn in range(max_turns):
        text, pt, ct = chat_fn(messages, max_tokens)
        transcript.prompt_tokens += pt
        transcript.completion_tokens += ct
        transcript.turns = turn + 1
        command = extract_shell(text)
        if not command:
            transcript.final_text = text
            return transcript
        transcript.commands.append(command)
        try:
            result = exec_fn(command)
        except Exception as exc:  # tool errors feed back, never crash the loop
            result = ExecResult(125, "", f"exec failed: {exc}")
        messages.append({"role": "assistant", "content": text})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"shell exit={result.exit_code} "
                    f"timed_out={result.timed_out}\n"
                    f"stdout:\n{result.stdout[-4000:]}\n"
                    f"stderr:\n{result.stderr[-4000:]}\n"
                    "Continue with another ```shell block, or give the final answer."
                ),
            }
        )
    transcript.final_text = text
    transcript.capped = True
    return transcript


@dataclass
class DockerShell:
    """Long-lived container for shell-agent trials."""

    image: str
    workdir: str = "/w"
    container_id: str = ""

    def start(self, timeout_secs: float = 300.0) -> DockerShell:
        name = f"bh-agent-{uuid.uuid4().hex[:8]}"
        proc = run_local(
            [
                "docker",
                "run",
                "-d",
                "--rm",
                "--name",
                name,
                "-w",
                self.workdir,
                self.image,
                "sleep",
                "3600",
            ],
            timeout_secs=timeout_secs,
        )
        if proc.exit_code != 0:
            raise RuntimeError(f"docker run failed: {proc.stderr[-400:]}")
        self.container_id = name
        return self

    def exec(self, command: str, timeout_secs: float = 300.0) -> ExecResult:
        if not self.container_id:
            raise RuntimeError("container not started")
        return run_local(
            ["docker", "exec", self.container_id, "bash", "-lc", command],
            timeout_secs=timeout_secs,
        )

    def stop(self) -> None:
        if self.container_id:
            run_local(["docker", "rm", "-f", self.container_id], timeout_secs=60.0)
            self.container_id = ""

    def __enter__(self) -> DockerShell:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


AGENT_SYSTEM = (
    "You are an autonomous coding agent in a Linux container. Act by replying "
    "with shell commands, one fenced block per turn:\n```shell\n<commands>\n```\n"
    "Work step by step: explore, write files with heredocs, run tests, fix "
    "failures. When the task is complete, reply with plain text (no fence) "
    "summarizing what you did."
)


def docker_shell_loop(
    client,
    model: str,
    image: str,
    user_prompt: str,
    max_turns: int = 20,
    max_tokens: int = 4096,
    exec_timeout_secs: float = 300.0,
) -> tuple[AgentTranscript, DockerShell]:
    """Full loop against a real container. Caller must stop() the shell
    (or collect the verifier result first — shell stays up on return)."""
    shell = DockerShell(image).start()

    def chat_fn(messages, cap):
        resp = client.chat(messages, model=model, max_tokens=cap)
        return (client.extract_text(resp), *client.extract_usage(resp))

    transcript = run_shell_loop(
        chat_fn,
        lambda cmd: shell.exec(cmd, exec_timeout_secs),
        AGENT_SYSTEM,
        user_prompt,
        max_turns,
        max_tokens,
    )
    return transcript, shell
