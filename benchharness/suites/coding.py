"""Coding suites. Wiring tiers:

wired    = dataset loader + runnable scorer today
scaffold = dataset/repo pointer recorded, full oracle loop lands next iteration
           (tasks still enumerable where the dataset is importable)

SWE-family adapters share _SweBase: HF dataset load, patch-format prompt,
patch extraction, and official docker FAIL_TO_PASS grading when the
swebench package + daemon are available (heuristic patch credit otherwise).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import ClassVar

from benchharness.schema import Score
from benchharness.suites.base import Requirement, SuiteAdapter, Task, strip_thinking
from benchharness.suites.reasoning import _load_hf_dataset

SWE_SYSTEM = (
    "You are fixing a GitHub issue in the repository below. Think briefly "
    "(under 200 words of reasoning), then output ONLY a unified diff patch "
    "inside a single ```diff fenced code block that resolves the issue. "
    "Start the patch with 'diff --git' lines. No explanation outside the patch."
)


def extract_patch(text: str) -> str:
    m = re.search(r"```diff\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Truncation-tolerant: unterminated fence runs to end of output.
    m = re.search(r"```(?:diff|patch|udiff)\s*(.*)$", text, re.DOTALL | re.IGNORECASE)
    if m and ("diff --git" in m.group(1) or "--- a/" in m.group(1)):
        return m.group(1).split("```")[0].strip()
    m = re.search(r"(diff --git .*|--- a/.*)", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def touched_files(patch: str) -> set[str]:
    return set(
        re.findall(r"^(?:diff --git a/|--- a/|\+\+\+ b/)(\S+)", patch, re.MULTILINE)
    )


class _SweBase(SuiteAdapter):
    category = "coding"
    dataset: str = ""
    dataset_config: str | None = None
    status = "wired"
    # Thinking models reason for thousands of tokens before the patch;
    # ornith-35b was still mid-analysis at 16k. 32k gives deep reasoning
    # plus the answer room (--max-tokens overrides).
    max_tokens = 32768
    # Datasets the installed swebench package can grade officially.
    # ScaleAI/SWE-bench_Pro uses a different image/eval layout; heuristic
    # grading until its driver is validated.
    docker_datasets = frozenset(
        {
            "princeton-nlp/SWE-bench_Verified",
            "SWE-bench/SWE-bench_Multilingual",
        }
    )

    def prepare(self, workdir: Path) -> None:
        self._workdir = workdir / "swe-reports"
        self._workdir.mkdir(parents=True, exist_ok=True)

    SMOKE: ClassVar[list] = [
        (
            "smoke-1",
            "Repo: demo/pkg. Issue: `add(a, b)` returns a-b instead of a+b.",
            "diff --git a/demo/pkg.py b/demo/pkg.py\n- return a-b\n+ return a+b",
        ),
    ]

    def requirements(self):
        return [
            Requirement(
                "pip", "datasets", "HF dataset loader (else smoke samples)", soft=True
            )
        ]

    def tasks(self, limit=None):
        rows = (
            _load_hf_dataset(self.dataset, self.dataset_config)
            if self.dataset
            else None
        )
        out: list[Task] = []
        if rows is not None:
            for row in rows:
                repo = row.get("repo", "")
                problem = row.get("problem_statement", "")
                hints = row.get("hints_text", "") or ""
                prompt = f"Repository: {repo}\n\nIssue:\n{problem}"
                if hints:
                    prompt += f"\n\nHints:\n{hints}"
                prompt += "\n\nProvide the fix as a unified diff patch."
                f2p = row.get("FAIL_TO_PASS", row.get("fail_to_pass", ""))
                p2p = row.get("PASS_TO_PASS", row.get("pass_to_pass", ""))
                out.append(
                    Task(
                        task_id=str(row.get("instance_id", len(out))),
                        prompt=prompt,
                        reference=str(row.get("patch", "")),
                        metadata={
                            "repo": repo,
                            "base_commit": row.get("base_commit", ""),
                            "fail_to_pass": str(f2p),
                            "pass_to_pass": str(p2p),
                        },
                        system=SWE_SYSTEM,
                    )
                )
        else:
            for tid, prompt, ref in self.SMOKE:
                out.append(
                    Task(task_id=tid, prompt=prompt, reference=ref, system=SWE_SYSTEM)
                )
        return out[:limit] if limit else out

    def score(self, output, task):
        patch = extract_patch(strip_thinking(output))
        if not patch:
            return Score(passed=False, details="no patch extracted")
        docked = self._docker_grade(task, patch)
        if docked is not None:
            return docked
        return self._heuristic_score(task, patch)

    def _docker_grade(self, task: Task, patch: str) -> Score | None:
        """Official docker grading, or None when unavailable/inapplicable."""
        if self.dataset not in self.docker_datasets:
            return None
        try:
            from benchharness.swe_eval import (
                docker_grading_available,
                grade_with_docker,
            )
        except Exception:
            return None
        if not docker_grading_available():
            return None
        # prepare() runs before scoring in real runs; unit tests that skip
        # it (and lack the swebench/docker stack) stay on heuristics.
        workdir = getattr(self, "_workdir", None)
        if workdir is None:
            return None
        try:
            grade = grade_with_docker(self.dataset, task.task_id, patch, workdir)
        except Exception as exc:
            return self._heuristic_score(
                task, patch, prefix=f"docker eval crashed ({exc}); heuristic: "
            )
        if "crashed" in grade.details or "unreadable" in grade.details:
            return self._heuristic_score(
                task, patch, prefix=f"{grade.details}; heuristic: "
            )
        return Score(
            passed=grade.resolved,
            score=1.0 if grade.resolved else 0.0,
            details=grade.details,
        )

    def _heuristic_score(self, task: Task, patch: str, prefix: str = "") -> Score:
        if task.reference and patch.strip() == task.reference.strip():
            return Score(passed=True, score=1.0, details=prefix + "exact patch match")
        ref_files = touched_files(task.reference)
        got_files = touched_files(patch)
        if ref_files and got_files & ref_files:
            overlap = len(got_files & ref_files) / len(ref_files)
            return Score(
                passed=False,
                score=0.5 * overlap,
                details=prefix + f"partial: touches "
                f"{sorted(got_files & ref_files)} "
                "(heuristic; docker eval unavailable)",
            )
        return Score(
            passed=False,
            score=0.1 if len(patch) > 20 else 0.0,
            details=prefix + "patch extracted but touches no gold files",
        )


class SweVerifiedAdapter(_SweBase):
    name = "swe-verified"
    description = "SWE-bench Verified (princeton-nlp/SWE-bench_Verified), 500 instances"
    dataset = "princeton-nlp/SWE-bench_Verified"
    source = "princeton-nlp/SWE-bench_Verified"


class SweProAdapter(SuiteAdapter):
    """SWE-bench Pro V2: 642 Harbor tasks (v2/tasks in scaleapi/SWE-bench_Pro-os).

    Runs `harbor run -p v2/tasks` per task (default agent terminus-2 routed
    to LM Studio; BENCH_HARBOR_AGENT=oracle for LM-free smoke). The full
    locked protocol (offline agent + patch_replay re-grade) is future work;
    single-trial rewards are reported.
    """

    name = "swe-pro"
    category = "coding"
    description = "SWE-bench Pro V2 (Harbor tasks, 642 instances)"
    source = "github.com/scaleapi/SWE-bench_Pro-os v2/tasks + ScaleAI/SWE-bench_Pro"
    status = "wired"
    harbor_agent = "terminus-2"
    task_timeout_secs = 2400.0  # above harbor_timeout_secs + image pulls
    repo_url = "https://github.com/scaleapi/SWE-bench_Pro-os"

    def requirements(self):
        return [
            Requirement("cli", "harbor", "Harbor runner (uv tool install harbor)"),
            Requirement("cli", "docker", "per-task images (ghcr.io/scaleapi)"),
            Requirement("cli", "git", "clone the task repo"),
        ]

    def _repo_dir(self) -> Path:
        import os as _os

        override = _os.environ.get("BENCH_SWE_PRO_REPO", "").strip()
        if override:
            return Path(override)
        from benchharness.harbor_driver import default_cache_dir

        return default_cache_dir() / "swe-bench_Pro-os"

    def _tasks_dir(self) -> Path:
        return self._repo_dir() / "v2" / "tasks"

    def prepare(self, workdir: Path) -> None:
        from benchharness.harbor_driver import require_docker_daemon
        from benchharness.sandbox import run_local

        require_docker_daemon()
        repo = self._repo_dir()
        tasks_dir = self._tasks_dir()
        if ((tasks_dir / "hard51_ids.txt").is_file() or tasks_dir.is_dir()) and any(
            tasks_dir.iterdir()
        ):
            return
        repo.parent.mkdir(parents=True, exist_ok=True)
        if repo.is_dir():
            fetched = run_local(
                ["git", "-C", str(repo), "fetch", "--depth", "1", "origin", "main"],
                timeout_secs=300.0,
            )
            if fetched.exit_code == 0:
                run_local(
                    ["git", "-C", str(repo), "reset", "--hard", "origin/main"],
                    timeout_secs=120.0,
                )
        else:
            proc = run_local(
                ["git", "clone", "--depth", "1", self.repo_url, str(repo)],
                timeout_secs=600.0,
            )
            if proc.exit_code != 0:
                raise RuntimeError(f"clone failed: {proc.stderr[-400:]}")
        if not self._tasks_dir().is_dir():
            raise RuntimeError(f"v2/tasks missing after clone: {repo}")

    def tasks(self, limit=None):
        from benchharness.harbor_driver import list_tasks

        names = list_tasks(self._tasks_dir())
        out = [
            Task(
                task_id=n,
                prompt=f"SWE-bench Pro task: {n}",
                reference="",
                metadata={"harbor_task": n},
            )
            for n in names
        ]
        return out[:limit] if limit else out

    def score(self, output, task):
        return Score(passed=False, details="swe-pro Harbor trial (see run_external)")

    def run_external(self, task, ctx):
        import os as _os

        from benchharness.harbor_driver import (
            build_run_command,
            parse_job_dir,
            run_job,
        )

        config = ctx.get("config")
        base_url = getattr(config, "base_url", "http://127.0.0.1:1234/v1")
        timeout = float(getattr(config, "harbor_timeout_secs", 1800.0))
        memory = getattr(config, "harbor_memory_policy", "ignore")
        model = str(ctx.get("model", ""))
        workdir: Path = ctx["workdir"]
        jobs_dir = workdir / "harbor-jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        job_name = f"swe-pro-{task.task_id}".replace("/", "_")[:80]

        agent = _os.environ.get("BENCH_HARBOR_AGENT", self.harbor_agent)
        agent_kwargs: dict[str, str] = {}
        agent_env: dict[str, str] = {}
        harbor_model = model
        if agent == "terminus-2":
            agent_kwargs["api_base"] = base_url
            if not model.startswith("openai/"):
                harbor_model = f"openai/{model}"
            agent_env["OPENAI_API_KEY"] = getattr(config, "api_key", "lm-studio")

        task_dir = task.metadata.get("harbor_task", task.task_id)
        from benchharness.harbor_driver import ensure_image, task_image

        image = task_image(self._tasks_dir() / task_dir)
        if image:
            ensure_image(image)
        cmd = build_run_command(
            "swe-bench-pro-v2",
            agent,
            harbor_model,
            jobs_dir,
            job_name,
            include_task=task_dir,  # local -p datasets match bare dir names
            n_tasks=1,
            agent_kwargs=agent_kwargs,
            agent_env=agent_env,
            memory_policy=memory,
            dataset_path=self._tasks_dir(),
        )
        # In-process agents (terminus-2 via litellm) read the subprocess
        # env, not the container env: merge agent_env into the child.
        returncode, tail = run_job(cmd, timeout, {**_os.environ, **agent_env})
        outcomes = parse_job_dir(jobs_dir / job_name)
        if not outcomes:
            return tail, Score(
                passed=False, score=0.0, details=f"no trial results (rc={returncode})"
            )
        oc = outcomes[0]
        if oc.error:
            return tail, Score(
                passed=False, score=0.0, details=f"trial error: {oc.error}"
            )
        return tail, Score(
            passed=oc.passed,
            score=oc.score,
            details=f"rewards={oc.rewards} ({oc.seconds:.0f}s)",
        )


class SweMultilingualAdapter(_SweBase):
    name = "swe-multilingual"
    description = "SWE-bench Multilingual (SWE-bench/SWE-bench_Multilingual)"
    dataset = "SWE-bench/SWE-bench_Multilingual"
    source = "SWE-bench/SWE-bench_Multilingual"


def _load_deepswe_tasks_dir():
    """Local path of the evalscope/deep-swe ModelScope snapshot's tasks/ dir
    (ungated mirror of the gated datacurve/deep-swe). None when offline."""
    try:
        from modelscope import snapshot_download  # type: ignore
    except Exception:
        return None
    try:
        root = Path(snapshot_download("evalscope/deep-swe", repo_type="dataset"))
        tasks_dir = root / "tasks"
        return tasks_dir if tasks_dir.is_dir() else None
    except Exception:
        return None


class DeepSweAdapter(SuiteAdapter):
    """DeepSWE via EvalScope's Pier integration (evalscope/deep-swe mirror,
    117 Harbor-format tasks). The Pier agent drives the repo task in Docker
    and the binary verifier reward is exposed as acc. Agent + judge both
    route to LM Studio (OpenAI-compatible, litellm model class)."""

    name = "deepswe"
    category = "coding"
    description = "DeepSWE (EvalScope Pier agent, 113 tasks, verifier acc)"
    source = "evalscope/deep-swe on ModelScope via EvalScope deep_swe"
    status = "wired"
    task_timeout_secs = 6000.0  # above evalscope_timeout_secs

    def requirements(self):
        return [
            Requirement("pip", "modelscope", "ModelScope snapshot", soft=True),
            Requirement(
                "evalscope",
                "deep_swe",
                "isolated EvalScope venv with Pier (deep_swe extra)",
            ),
            Requirement("cli", "docker", "Pier task envs"),
        ]

    def tasks(self, limit=None):
        tasks_dir = _load_deepswe_tasks_dir()
        if tasks_dir is None:
            return [
                Task(
                    task_id="missing-data",
                    prompt="",
                    reference="",
                    metadata={
                        "skip_reason": "evalscope/deep-swe snapshot "
                        "unavailable; pip install modelscope + network"
                    },
                )
            ]
        out = []
        for child in sorted(tasks_dir.iterdir()):
            if not (child.is_dir() and (child / "task.toml").is_file()):
                continue
            instruction = child / "instruction.md"
            prompt = (
                instruction.read_text(encoding="utf-8")[:4000]
                if instruction.is_file()
                else f"DeepSWE task: {child.name}"
            )
            out.append(
                Task(
                    task_id=child.name,
                    prompt=prompt,
                    reference="",
                    metadata={"tasks_dir": str(tasks_dir)},
                )
            )
        return out[:limit] if limit else out

    def score(self, output, task):
        return Score(passed=False, details="deepswe EvalScope trial (see run_external)")

    def run_external(self, task, ctx):
        import os as _os

        from benchharness.evalscope_driver import run_one

        config = ctx.get("config")
        model = str(ctx.get("model", ""))
        trials = int(_os.environ.get("BENCH_DEEPSWE_TRIALS", "1"))
        timeout = float(getattr(config, "evalscope_timeout_secs", 5400.0))
        oc = run_one(
            "deep_swe",
            task.task_id,
            model=model,
            api_base=getattr(config, "base_url", "http://127.0.0.1:1234/v1"),
            api_key=getattr(config, "api_key", "lm-studio"),
            trials=trials,
            work_dir=ctx["workdir"],
            timeout_secs=timeout,
        )
        output = f"deep_swe trials={trials} trace={oc.trace_path}"
        if oc.error:
            return output, Score(
                passed=False, score=0.0, details=f"{oc.details} [{oc.error}]"
            )
        return output, Score(passed=oc.passed, score=oc.score, details=oc.details)


class FrontierBenchAdapter(SuiteAdapter):
    """Frontier-Bench v0.1: Anthropic-reported agentic coding numbers (Opus 5
    at 43.3%) with no public dataset or harness as of iteration 14
    (Harbor Hub re-checked: 340 datasets, no frontier entry; TB4 line
    continues via ryanmarten/tb4-preview, not Frontier itself).
    frontierbench.ai redirects to tbench.ai. Stays scaffold; re-check
    the Harbor registry + tbench.ai each iteration."""

    name = "frontier-bench"
    category = "coding"
    description = "Frontier-Bench v0.1 (no public dataset found yet)"
    source = "unresolved: no public dataset/harness located"
    status = "scaffold"

    def tasks(self, limit=None):
        return [
            Task(
                task_id="no-public-source",
                prompt="",
                reference="",
                metadata={
                    "skip_reason": "no public Frontier-Bench "
                    "dataset/harness found; rerun research"
                },
            )
        ]

    def score(self, output, task):
        return Score(passed=False, details="scaffold: no public source")


class Nl2RepoAdapter(SuiteAdapter):
    """NL2Repo (AweAI-Team/AweAgent-Meta-NL2Repo): build a repo from an NL spec.

    Custom shell-agent loop + per-task verify_cmd grading (proven live:
    20-turn loop, verify exit mapping, transcript.json per task).
    The official nl2repobench/nl2repobench Harbor set (104 tasks) is
    NOT usable: every tester sidecar image sits on a private GCP
    Artifact Registry (Unauthenticated 403). Revisit if the publisher
    opens the registry; migration is a ~20-line _TerminalBenchBase
    subclass (dataset_default + prompt_prefix).
    """

    name = "nl2repo"
    category = "coding"
    description = "NL2Repo (AweAI-Team/AweAgent-Meta-NL2Repo, shell agent + verify)"
    source = "AweAI-Team/AweAgent-Meta-NL2Repo"
    status = "wired"
    task_timeout_secs = 1800.0  # 20-turn shell-agent loop + verifier
    max_tokens = 16384
    agent_turns = 20

    def requirements(self):
        return [
            Requirement("pip", "datasets", "HF loader", soft=True),
            Requirement("cli", "docker", "per-task evaluation images"),
        ]

    def tasks(self, limit=None):
        rows = _load_hf_dataset("AweAI-Team/AweAgent-Meta-NL2Repo")
        if rows is None:
            return [
                Task(
                    task_id="missing-data",
                    prompt="",
                    reference="",
                    metadata={
                        "skip_reason": "NL2Repo dataset offline; "
                        "install datasets + network"
                    },
                )
            ]
        out = [
            Task(
                task_id=str(r.get("instance_id", i)),
                prompt=str(r.get("start_instruction", "")),
                reference="",
                metadata={
                    "package": r.get("package_name", ""),
                    "verify_cmd": str(r.get("verify_cmd", "")),
                    "image": r.get("evaluation_image", ""),
                },
            )
            for i, r in enumerate(rows)
        ]
        return out[:limit] if limit else out

    def score(self, output, task):
        # Unused: run_external() drives the shell-agent loop + verifier.
        return Score(passed=False, details="nl2repo agent trial (see run_external)")

    def run_external(self, task, ctx):
        import json as _json

        from benchharness.agent_loop import docker_shell_loop

        image = task.metadata.get("image", "")
        verify_cmd = task.metadata.get("verify_cmd", "")
        if not image:
            raise RuntimeError("NL2Repo task has no evaluation_image")
        client = ctx.get("client")
        model = str(ctx.get("model", ""))
        transcript, shell = docker_shell_loop(
            client,
            model,
            image,
            task.prompt,
            max_turns=self.agent_turns,
            max_tokens=self.max_tokens,
        )
        try:
            verify = shell.exec(verify_cmd, timeout_secs=600.0) if verify_cmd else None
        finally:
            shell.stop()
        workdir = Path(ctx.get("workdir", "."))
        workdir.mkdir(parents=True, exist_ok=True)
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", task.task_id)[:60]
        (workdir / f"transcript-{safe_id}.json").write_text(
            _json.dumps(
                {
                    "turns": transcript.turns,
                    "capped": transcript.capped,
                    "commands": transcript.commands,
                    "final_text": transcript.final_text[:2000],
                    "prompt_tokens": transcript.prompt_tokens,
                    "completion_tokens": transcript.completion_tokens,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if verify is None:
            return "", Score(
                passed=False,
                details=f"no verify_cmd; agent ran {transcript.turns} turns",
                prompt_tokens=transcript.prompt_tokens,
                completion_tokens=transcript.completion_tokens,
            )
        ok = verify.exit_code == 0
        cmd_lines = "\n".join(f"$ {c}" for c in transcript.commands[-8:])
        tail = (verify.stdout + "\n" + verify.stderr)[-600:]
        return f"{cmd_lines}\n--- verify ---\n{tail}", Score(
            passed=ok,
            score=1.0 if ok else 0.0,
            details=f"verify exit={verify.exit_code} after {transcript.turns} "
            f"turns{' (capped)' if transcript.capped else ''}",
            prompt_tokens=transcript.prompt_tokens,
            completion_tokens=transcript.completion_tokens,
        )


def _keywords(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9_]{4,}", text.lower())
    stop = {
        "that",
        "this",
        "with",
        "from",
        "have",
        "will",
        "about",
        "into",
        "your",
        "what",
        "when",
        "where",
        "which",
        "their",
        "there",
        "code",
        "file",
        "files",
        "function",
        "using",
        "used",
        "such",
        "than",
        "then",
    }
    return {w for w in words if w not in stop}


class SweAtlasQnaAdapter(SuiteAdapter):
    """SWE Atlas QnA (ScaleAI/SWE-Atlas-QnA): 124 deep code-comprehension Qs.

    Scores rubric-keyword recall: fraction of rubric keywords covered by the
    answer (+ LLM rubric judge via BENCH_JUDGE=1).
    """

    name = "swe-atlas-qna"
    category = "coding"
    description = "SWE Atlas QnA (ScaleAI/SWE-Atlas-QnA, rubric-keyword recall)"
    source = "ScaleAI/SWE-Atlas-QnA (124 tasks, 11 repos)"
    status = "wired"
    max_tokens = 4096

    SMOKE: ClassVar[list] = [
        (
            "smoke-1",
            (
                "Repo utils (python): where is the retry helper defined and what "
                "does it do?"
            ),
            "lib/retry.py defines retry() with backoff",
            "retry backoff lib",
        )
    ]

    def requirements(self):
        return [Requirement("pip", "datasets", "HF loader (else smoke)", soft=True)]

    def tasks(self, limit=None):
        rows = _load_hf_dataset("ScaleAI/SWE-Atlas-QnA")
        out: list[Task] = []
        if rows is not None:
            for r in rows:
                out.append(
                    Task(
                        task_id=str(r.get("task_id", len(out))),
                        prompt=f"Repository: {r.get('repository_url', '')} "
                        f"@{r.get('repository_base_commit', '')} "
                        f"({r.get('language', '')})\n\n"
                        f"{r.get('prompt', '')}",
                        reference=str(r.get("reference_answer", "")),
                        metadata={
                            "rubric": str(r.get("rubric", "")),
                            "category": str(r.get("category", "")),
                        },
                    )
                )
        else:
            for tid, prompt, ref, rubric in self.SMOKE:
                out.append(
                    Task(
                        task_id=tid,
                        prompt=prompt,
                        reference=ref,
                        metadata={"rubric": rubric},
                    )
                )
        return out[:limit] if limit else out

    def score(self, output, task):
        answer = strip_thinking(output)
        rubric_keys = _keywords(task.metadata.get("rubric", "") or task.reference)
        if not rubric_keys:
            return Score(passed=False, details="no rubric keywords")
        hits = {k for k in rubric_keys if k in answer.lower()}
        recall = len(hits) / len(rubric_keys)
        return Score(
            passed=recall >= 0.6,
            score=recall,
            details=f"rubric_recall={len(hits)}/{len(rubric_keys)} "
            "(BENCH_JUDGE=1 for LLM rubric judge)",
        )

    def score_with_client(self, output, task, client, model):
        from benchharness.judge import judge_correct

        rubric = task.metadata.get("rubric", "")
        gold = task.reference + (f"\nRubric: {rubric}" if rubric else "")
        verdict, _ = judge_correct(
            client, model, task.prompt, gold, strip_thinking(output)
        )
        if verdict is None:
            return None
        return Score(
            passed=verdict, score=1.0 if verdict else 0.0, details="LLM rubric judge"
        )


class _TerminalBenchBase(SuiteAdapter):
    """Terminal-Bench 2.1 through Harbor, one `harbor run` trial per task.

    Both adapters share the task set; the `--agent` harness differs.
    terminus-2 routes to local LM Studio via `--ak api_base=...`; the
    claude-code agent shells to the Claude Code CLI and needs
    ANTHROPIC_API_KEY (or an ANTHROPIC_BASE_URL shim).
    """

    category = "coding"
    status = "wired"
    harbor_agent = ""
    task_timeout_secs = 2400.0  # above harbor_timeout_secs + image pulls

    def requirements(self):
        reqs = [
            Requirement("cli", "harbor", "Harbor runner (uv tool install harbor)"),
            Requirement("cli", "docker", "task containers + oracle"),
        ]
        if self.harbor_agent == "claude-code":
            reqs.append(
                Requirement("env", "ANTHROPIC_API_KEY", "Claude Code CLI credential")
            )
        if self.harbor_agent == "hermes":
            reqs.append(Requirement("cli", "hermes", "Hermes Agent CLI"))
        return reqs

    def _dataset(self) -> str:
        import os as _os

        # Registry form is org/name; legacy terminal-bench@2.0 still works
        # via TB_DATASET override.
        return _os.environ.get("TB_DATASET", "terminal-bench/terminal-bench-2-1")

    def prepare(self, workdir: Path) -> None:
        from benchharness.harbor_driver import ensure_dataset, require_docker_daemon

        require_docker_daemon()
        ensure_dataset(self._dataset())

    def tasks(self, limit=None):
        from benchharness.harbor_driver import (
            dataset_dir_name,
            default_cache_dir,
            list_tasks,
        )

        names = list_tasks(default_cache_dir() / dataset_dir_name(self._dataset()))
        out = [
            Task(
                task_id=n,
                prompt=f"Terminal-Bench task: {n}",
                reference="",
                metadata={"harbor_task": n, "harbor_dataset": self._dataset()},
            )
            for n in names
        ]
        return out[:limit] if limit else out

    def score(self, output, task):
        # Unused: run_external() bypasses the chat flow entirely.
        return Score(passed=False, details="harbor trial (see run_external)")

    def run_external(self, task, ctx):
        import os as _os
        import sys as _sys

        from benchharness.harbor_driver import (
            build_run_command,
            parse_job_dir,
            run_job,
        )

        config = ctx.get("config")
        base_url = getattr(config, "base_url", "http://127.0.0.1:1234/v1")
        timeout = float(getattr(config, "harbor_timeout_secs", 1800.0))
        memory = getattr(config, "harbor_memory_policy", "ignore")
        model = str(ctx.get("model", ""))
        workdir: Path = ctx["workdir"]
        jobs_dir = workdir / "harbor-jobs"
        jobs_dir.mkdir(parents=True, exist_ok=True)
        try:
            jobs_dir.resolve().relative_to(Path.home().resolve())
        except ValueError:
            print(
                "warning: harbor jobs outside $HOME are invisible to Colima "
                "bind mounts; rewards will not download. Keep --out under $HOME.",
                file=_sys.stderr,
            )
        job_name = f"{self.name}-{task.task_id}".replace("/", "_")[:80]

        # BENCH_HARBOR_AGENT=oracle runs golden solutions (LM-free smoke test).
        agent = _os.environ.get("BENCH_HARBOR_AGENT", self.harbor_agent)
        agent_kwargs: dict[str, str] = {}
        agent_env: dict[str, str] = {}
        harbor_model = model
        if agent == "terminus-2":
            agent_kwargs["api_base"] = base_url
            if not model.startswith("openai/"):
                harbor_model = f"openai/{model}"
            agent_env["OPENAI_API_KEY"] = getattr(config, "api_key", "lm-studio")
        elif agent == "claude-code":
            if _os.environ.get("ANTHROPIC_API_KEY"):
                agent_env["ANTHROPIC_API_KEY"] = _os.environ["ANTHROPIC_API_KEY"]

        dataset = self._dataset()
        bare = task.metadata.get("harbor_task", task.task_id)
        # org/name datasets need qualified filters (terminal-bench/<task>);
        # legacy name@version datasets take bare names.
        if "/" in dataset and "/" not in bare:
            include = f"{dataset.split('@')[0].split('/')[0]}/{bare}"
        else:
            include = bare
        cmd = build_run_command(
            dataset,
            agent,
            harbor_model,
            jobs_dir,
            job_name,
            include_task=include,
            n_tasks=1,
            agent_kwargs=agent_kwargs,
            agent_env=agent_env,
            memory_policy=memory,
        )
        # In-process agents (terminus-2 via litellm) read the subprocess
        # env, not the container env: merge agent_env into the child.
        returncode, tail = run_job(cmd, timeout, {**_os.environ, **agent_env})
        outcomes = parse_job_dir(jobs_dir / job_name)
        if not outcomes:
            return tail, Score(
                passed=False, score=0.0, details=f"no trial results (rc={returncode})"
            )
        oc = outcomes[0]
        if oc.error:
            return tail, Score(
                passed=False, score=0.0, details=f"trial error: {oc.error}"
            )
        return tail, Score(
            passed=oc.passed,
            score=oc.score,
            details=f"rewards={oc.rewards} ({oc.seconds:.0f}s)",
        )


class TbTerminusAdapter(_TerminalBenchBase):
    name = "tb-terminus"
    description = "Terminal-Bench 2.1 via Terminus-2 harness (Harbor, LM Studio-routed)"
    source = "harbor run --dataset terminal-bench-2-1 --agent terminus-2"
    harbor_agent = "terminus-2"


class TbClaudeAdapter(_TerminalBenchBase):
    name = "tb-claude"
    description = "Terminal-Bench 2.1 via Claude Code harness (Harbor, needs API key)"
    source = "harbor run --dataset terminal-bench-2-1 --agent claude-code"
    harbor_agent = "claude-code"


class TbHermesAdapter(_TerminalBenchBase):
    """Terminal-Bench 2.1 via the Hermes Agent harness (Harbor `--agent hermes`).

    Model routing is Hermes-native: point the hermes CLI at LM Studio first
    (`hermes model` / provider config), then the trials use it. No api_base
    kwarg exists on this agent.
    """

    name = "tb-hermes"
    description = "Terminal-Bench 2.1 via Hermes Agent harness (Harbor)"
    source = "harbor run --dataset terminal-bench-2-1 --agent hermes"
    harbor_agent = "hermes"
