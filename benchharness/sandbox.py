"""Command sandbox: local subprocess execution plus docker execution.

- ``run_local``: runs argv with timeout, captures stdout/stderr/exit.
- ``run_docker``: ``docker run --rm --network=none -v workdir:/w <image>``
  when docker is available; raises SandboxUnavailable otherwise.
- ``run_python_snippet``: backs the hle-tools ``run_python`` tool loop.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


class SandboxUnavailable(RuntimeError):
    pass


@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


def have_docker() -> bool:
    return shutil.which("docker") is not None


def run_local(
    argv: list[str],
    cwd: Path | None = None,
    timeout_secs: float = 60.0,
    env: dict | None = None,
) -> ExecResult:
    try:
        proc = subprocess.run(
            argv, cwd=cwd, capture_output=True, text=True,
            timeout=timeout_secs, env=env,
        )
        return ExecResult(proc.returncode, proc.stdout[-8000:], proc.stderr[-8000:])
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        err = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        return ExecResult(124, out[-8000:], err[-8000:], timed_out=True)
    except OSError as exc:
        return ExecResult(127, "", str(exc))


def run_docker(
    image: str,
    argv: list[str],
    workdir: Path,
    timeout_secs: float = 300.0,
    network: str = "none",
) -> ExecResult:
    if not have_docker():
        raise SandboxUnavailable("docker CLI not found")
    workdir.mkdir(parents=True, exist_ok=True)
    cmd = ["docker", "run", "--rm", f"--network={network}",
           "-v", f"{workdir.resolve()}:/w", "-w", "/w", image, *argv]
    return run_local(cmd, timeout_secs=timeout_secs)


def run_python_snippet(code: str, timeout_secs: float = 30.0) -> ExecResult:
    """Execute a model-provided snippet in a subprocess (stdout-capped)."""
    return run_local([sys.executable, "-c", code], timeout_secs=timeout_secs)
