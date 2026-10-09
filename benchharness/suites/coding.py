"""Coding suites. Wiring tiers:

wired    = dataset loader + runnable scorer today
scaffold = dataset/repo pointer recorded, full oracle loop lands next iteration
           (tasks still enumerable where the dataset is importable)

SWE-family adapters share _SweBase: HF dataset load, patch-format prompt,
and patch-extraction scoring. Docker-based FAIL_TO_PASS execution is the
documented iteration-2 upgrade; today a well-formed non-empty patch that
touches the gold files scores partial credit, exact-match scores full.
"""

from __future__ import annotations

import re
from pathlib import Path

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
    return set(re.findall(r"^(?:diff --git a/|--- a/|\+\+\+ b/)(\S+)", patch, re.MULTILINE))


class _SweBase(SuiteAdapter):
    category = "coding"
    dataset: str = ""
    dataset_config: str | None = None
    status = "wired"
    # Thinking models reason for thousands of tokens before the patch;
    # ornith-35b was still mid-analysis at 16k. 32k gives deep reasoning
    # plus the answer room (--max-tokens overrides).
    max_tokens = 32768

    SMOKE = [
        ("smoke-1",
         "Repo: demo/pkg. Issue: `add(a, b)` returns a-b instead of a+b.",
         "diff --git a/demo/pkg.py b/demo/pkg.py\n- return a-b\n+ return a+b"),
    ]

    def requirements(self):
        return [Requirement("pip", "datasets", "HF dataset loader (else smoke samples)", soft=True)]

    def tasks(self, limit=None):
        rows = _load_hf_dataset(self.dataset, self.dataset_config) if self.dataset else None
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
                out.append(Task(
                    task_id=str(row.get("instance_id", len(out))),
                    prompt=prompt,
                    reference=str(row.get("patch", "")),
                    metadata={"repo": repo,
                              "base_commit": row.get("base_commit", ""),
                              "fail_to_pass": str(f2p),
                              "pass_to_pass": str(p2p)},
                    system=SWE_SYSTEM,
                ))
        else:
            for tid, prompt, ref in self.SMOKE:
                out.append(Task(task_id=tid, prompt=prompt, reference=ref,
                                system=SWE_SYSTEM))
        return out[:limit] if limit else out

    def score(self, output, task):
        patch = extract_patch(strip_thinking(output))
        if not patch:
            return Score(passed=False, details="no patch extracted")
        ref_files = touched_files(task.reference)
        got_files = touched_files(patch)
        if task.reference and patch.strip() == task.reference.strip():
            return Score(passed=True, score=1.0, details="exact patch match")
        if ref_files and got_files & ref_files:
            overlap = len(got_files & ref_files) / len(ref_files)
            return Score(passed=False, score=0.5 * overlap,
                         details=f"partial: touches {sorted(got_files & ref_files)}; "
                                 "docker FAIL_TO_PASS eval lands in iteration 2")
        return Score(passed=False, score=0.1 if len(patch) > 20 else 0.0,
                     details="patch extracted but touches no gold files")


class SweVerifiedAdapter(_SweBase):
    name = "swe-verified"
    description = "SWE-bench Verified (princeton-nlp/SWE-bench_Verified), 500 instances"
    dataset = "princeton-nlp/SWE-bench_Verified"
    source = "princeton-nlp/SWE-bench_Verified"


class SweProAdapter(_SweBase):
    name = "swe-pro"
    description = "SWE-bench Pro (ScaleAI/SWE-bench_Pro), long-horizon tasks"
    dataset = "ScaleAI/SWE-bench_Pro"
    source = "ScaleAI/SWE-bench_Pro"


class SweMultilingualAdapter(_SweBase):
    name = "swe-multilingual"
    description = "SWE-bench Multilingual (SWE-bench/SWE-bench_Multilingual)"
    dataset = "SWE-bench/SWE-bench_Multilingual"
    source = "SWE-bench/SWE-bench_Multilingual"


class DeepSweAdapter(SuiteAdapter):
    """DeepSWE (datacurve/deep-swe): 113 long-horizon tasks, Harbor format.

    The dataset is gated (access request + HF_TOKEN) and grading needs the
    program verifiers + isolated envs; deferred until the Harbor driver lands.
    """

    name = "deepswe"
    category = "coding"
    description = "DeepSWE (datacurve/deep-swe, gated, Harbor-format verifiers)"
    source = "datacurve/deep-swe + github.com/datacurve-ai/deep-swe"
    status = "scaffold"

    def requirements(self):
        return [
            Requirement("pip", "datasets", "HF loader", soft=True),
            Requirement("cli", "harbor", "Harbor runner for TB/DeepSWE-style tasks"),
            Requirement("cli", "docker", "isolated task envs"),
            Requirement("env", "HF_TOKEN", "gated dataset access"),
        ]

    def tasks(self, limit=None):
        rows = _load_hf_dataset("datacurve/deep-swe")
        if rows is None:
            return [Task(task_id="gated", prompt="", reference="",
                         metadata={"skip_reason": "datacurve/deep-swe gated or "
                                   "offline; needs HF access + HF_TOKEN"})]
        out = [Task(task_id=str(r.get("id", i)), prompt=str(r.get("prompt", r)),
                    reference="", metadata={"skip_reason": "Harbor verifier loop "
                                             "lands in iteration 4"})
               for i, r in enumerate(rows)]
        return out[:limit] if limit else out

    def score(self, output, task):
        return Score(passed=False, details="scaffold: Harbor verifier loop pending")


class FrontierBenchAdapter(SuiteAdapter):
    """Frontier-Bench v0.1: agentic terminal coding (Anthropic-reported SOTA
    numbers exist, e.g. Opus 5 at 43.3%, but no public dataset or harness
    was found as of iteration 3). Stays scaffold until a source appears."""

    name = "frontier-bench"
    category = "coding"
    description = "Frontier-Bench v0.1 (no public dataset found yet)"
    source = "unresolved: no public dataset/harness located"
    status = "scaffold"

    def tasks(self, limit=None):
        return [Task(task_id="no-public-source", prompt="", reference="",
                     metadata={"skip_reason": "no public Frontier-Bench "
                               "dataset/harness found; rerun research"})]

    def score(self, output, task):
        return Score(passed=False, details="scaffold: no public source")


class Nl2RepoAdapter(SuiteAdapter):
    """NL2Repo (AweAI-Team/AweAgent-Meta-NL2Repo): build a repo from an NL spec.

    Task enumeration is real; the docker build + golden-test verify loop
    (verify_cmd in per-task images) lands in iteration 4, so tasks defer
    without burning model calls.
    """

    name = "nl2repo"
    category = "coding"
    description = "NL2Repo (AweAI-Team/AweAgent-Meta-NL2Repo, docker golden tests)"
    source = "AweAI-Team/AweAgent-Meta-NL2Repo"
    status = "scaffold"
    max_tokens = 16384

    def requirements(self):
        return [
            Requirement("pip", "datasets", "HF loader", soft=True),
            Requirement("cli", "docker", "per-task evaluation images"),
        ]

    def tasks(self, limit=None):
        rows = _load_hf_dataset("AweAI-Team/AweAgent-Meta-NL2Repo")
        if rows is None:
            return [Task(task_id="missing-data", prompt="", reference="",
                         metadata={"skip_reason": "NL2Repo dataset offline; "
                                   "install datasets + network"})]
        out = [Task(
            task_id=str(r.get("instance_id", i)),
            prompt=str(r.get("start_instruction", "")),
            reference="",
            metadata={"skip_reason": "docker build+verify loop lands in "
                                     "iteration 4",
                      "package": r.get("package_name", ""),
                      "verify_cmd": str(r.get("verify_cmd", "")),
                      "image": r.get("evaluation_image", "")},
        ) for i, r in enumerate(rows)]
        return out[:limit] if limit else out

    def score(self, output, task):
        return Score(passed=False, details="scaffold: docker verify loop pending")


def _keywords(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9_]{4,}", text.lower())
    stop = {"that", "this", "with", "from", "have", "will", "about", "into",
            "your", "what", "when", "where", "which", "their", "there", "code",
            "file", "files", "function", "using", "used", "such", "than", "then"}
    return {w for w in words if w not in stop}


class SweAtlasQnaAdapter(SuiteAdapter):
    """SWE Atlas QnA (ScaleAI/SWE-Atlas-QnA): 124 deep code-comprehension Qs.

    Scores rubric-keyword recall: fraction of rubric keywords covered by the
    answer. The official rubric LLM-judge lands in iteration 4.
    """

    name = "swe-atlas-qna"
    category = "coding"
    description = "SWE Atlas QnA (ScaleAI/SWE-Atlas-QnA, rubric-keyword recall)"
    source = "ScaleAI/SWE-Atlas-QnA (124 tasks, 11 repos)"
    status = "wired"
    max_tokens = 4096

    SMOKE = [("smoke-1",
              "Repo utils (python): where is the retry helper defined and what "
              "does it do?",
              "lib/retry.py defines retry() with backoff",
              "retry backoff lib")]

    def requirements(self):
        return [Requirement("pip", "datasets", "HF loader (else smoke)", soft=True)]

    def tasks(self, limit=None):
        rows = _load_hf_dataset("ScaleAI/SWE-Atlas-QnA")
        out: list[Task] = []
        if rows is not None:
            for r in rows:
                out.append(Task(
                    task_id=str(r.get("task_id", len(out))),
                    prompt=f"Repository: {r.get('repository_url', '')} "
                           f"@{r.get('repository_base_commit', '')} "
                           f"({r.get('language', '')})\n\n"
                           f"{r.get('prompt', '')}",
                    reference=str(r.get("reference_answer", "")),
                    metadata={"rubric": str(r.get("rubric", "")),
                              "category": str(r.get("category", ""))},
                ))
        else:
            for tid, prompt, ref, rubric in self.SMOKE:
                out.append(Task(task_id=tid, prompt=prompt, reference=ref,
                                metadata={"rubric": rubric}))
        return out[:limit] if limit else out

    def score(self, output, task):
        answer = strip_thinking(output)
        rubric_keys = _keywords(task.metadata.get("rubric", "") or task.reference)
        if not rubric_keys:
            return Score(passed=False, details="no rubric keywords")
        hits = {k for k in rubric_keys if k in answer.lower()}
        recall = len(hits) / len(rubric_keys)
        return Score(passed=recall >= 0.6, score=recall,
                     details=f"rubric_recall={len(hits)}/{len(rubric_keys)}; "
                             "LLM rubric judge lands in iteration 4")


class _TerminalBenchBase(SuiteAdapter):
    """Terminal-Bench 2.x through Harbor, one `harbor run` trial per task.

    Both adapters share the task set; the `--agent` harness differs.
    terminus-2 routes to local LM Studio via `--ak api_base=...`; the
    claude-code agent shells to the Claude Code CLI and needs
    ANTHROPIC_API_KEY (or an ANTHROPIC_BASE_URL shim).
    """

    category = "coding"
    status = "wired"
    harbor_agent = ""

    def requirements(self):
        reqs = [
            Requirement("cli", "harbor", "Harbor runner (uv tool install harbor)"),
            Requirement("cli", "docker", "task containers + oracle"),
        ]
        if self.harbor_agent == "claude-code":
            reqs.append(Requirement("env", "ANTHROPIC_API_KEY",
                                    "Claude Code CLI credential"))
        if self.harbor_agent == "hermes":
            reqs.append(Requirement("cli", "hermes", "Hermes Agent CLI"))
        return reqs

    def _dataset(self) -> str:
        import os as _os

        # 2.1 is not in the public Harbor registry yet; override with
        # TB_DATASET=terminal-bench@2.1 when it publishes.
        return _os.environ.get("TB_DATASET", "terminal-bench@2.0")

    def prepare(self, workdir: Path) -> None:
        from benchharness.harbor_driver import ensure_dataset

        ensure_dataset(self._dataset())

    def tasks(self, limit=None):
        from benchharness.harbor_driver import default_cache_dir, list_tasks

        from benchharness.harbor_driver import dataset_name_version

        name, _ = dataset_name_version(self._dataset())
        names = list_tasks(default_cache_dir() / name)
        out = [Task(task_id=n, prompt=f"Terminal-Bench task: {n}", reference="",
                    metadata={"harbor_task": n, "harbor_dataset": self._dataset()})
               for n in names]
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
            print("warning: harbor jobs outside $HOME are invisible to Colima "
                  "bind mounts; rewards will not download. Keep --out under $HOME.",
                  file=_sys.stderr)
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

        cmd = build_run_command(
            self._dataset(), agent, harbor_model, jobs_dir, job_name,
            include_task=task.metadata.get("harbor_task", task.task_id),
            n_tasks=1,
            agent_kwargs=agent_kwargs, agent_env=agent_env,
            memory_policy=memory,
        )
        returncode, tail = run_job(cmd, timeout)
        outcomes = parse_job_dir(jobs_dir / job_name)
        if not outcomes:
            return tail, Score(passed=False, score=0.0,
                               details=f"no trial results (rc={returncode})")
        oc = outcomes[0]
        if oc.error:
            return tail, Score(passed=False, score=0.0,
                               details=f"trial error: {oc.error}")
        return tail, Score(passed=oc.passed, score=oc.score,
                           details=f"rewards={oc.rewards} ({oc.seconds:.0f}s)")


class TbTerminusAdapter(_TerminalBenchBase):
    name = "tb-terminus"
    description = "Terminal-Bench 2.x via Terminus-2 harness (Harbor, LM Studio-routed)"
    source = "harbor run --dataset terminal-bench@2.x --agent terminus-2"
    harbor_agent = "terminus-2"


class TbClaudeAdapter(_TerminalBenchBase):
    name = "tb-claude"
    description = "Terminal-Bench 2.x via Claude Code harness (Harbor, needs API key)"
    source = "harbor run --dataset terminal-bench@2.x --agent claude-code"
    harbor_agent = "claude-code"


class TbHermesAdapter(_TerminalBenchBase):
    """Terminal-Bench 2.x via the Hermes Agent harness (Harbor `--agent hermes`).

    Model routing is Hermes-native: point the hermes CLI at LM Studio first
    (`hermes model` / provider config), then the trials use it. No api_base
    kwarg exists on this agent.
    """

    name = "tb-hermes"
    description = "Terminal-Bench 2.x via Hermes Agent harness (Harbor)"
    source = "harbor run --dataset terminal-bench@2.x --agent hermes"
    harbor_agent = "hermes"
