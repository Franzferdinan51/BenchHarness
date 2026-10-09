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

from benchharness.schema import Score
from benchharness.suites.base import Requirement, SuiteAdapter, Task
from benchharness.suites.reasoning import _load_hf_dataset

SWE_SYSTEM = (
    "You are fixing a GitHub issue in the repository below. Output ONLY a "
    "unified diff patch (```diff fenced block) that resolves the issue. "
    "No explanation outside the patch."
)


def extract_patch(text: str) -> str:
    m = re.search(r"```diff\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"(diff --git .*|--- a/.*)", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def touched_files(patch: str) -> set[str]:
    return set(re.findall(r"^(?:diff --git a/|--- a/|\+\+\+ b/)(\S+)", patch, re.MULTILINE))


class _SweBase(SuiteAdapter):
    category = "coding"
    dataset: str = ""
    dataset_config: str | None = None
    status = "wired"

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
                out.append(Task(
                    task_id=str(row.get("instance_id", len(out))),
                    prompt=prompt,
                    reference=str(row.get("patch", "")),
                    metadata={"repo": repo,
                              "base_commit": row.get("base_commit", ""),
                              "fail_to_pass": str(row.get("FAIL_TO_PASS", "")),
                              "pass_to_pass": str(row.get("PASS_TO_PASS", ""))},
                    system=SWE_SYSTEM,
                ))
        else:
            for tid, prompt, ref in self.SMOKE:
                out.append(Task(task_id=tid, prompt=prompt, reference=ref,
                                system=SWE_SYSTEM))
        return out[:limit] if limit else out

    def score(self, output, task):
        patch = extract_patch(output)
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
    description = "SWE-bench Pro (SWE-bench/SWE-bench_Pro)"
    dataset = "SWE-bench/SWE-bench_Pro"
    source = "SWE-bench/SWE-bench_Pro"
    status = "scaffold"  # dataset id tentative; verify against official release


class SweMultilingualAdapter(_SweBase):
    name = "swe-multilingual"
    description = "SWE-bench Multilingual (SWE-bench/SWE-bench_Multilingual)"
    dataset = "SWE-bench/SWE-bench_Multilingual"
    source = "SWE-bench/SWE-bench_Multilingual"


class DeepSweAdapter(_SweBase):
    name = "deepswe"
    description = "DeepSWE (long-horizon SWE tasks)"
    dataset = ""
    source = "DeepSWE official release (id TBD — verify)"
    status = "scaffold"


class FrontierBenchAdapter(SuiteAdapter):
    name = "frontier-bench"
    category = "coding"
    description = "Frontier-Bench v0.1 (frontier coding tasks)"
    source = "Frontier-Bench v0.1 official release (id TBD — verify)"
    status = "scaffold"

    def tasks(self, limit=None):
        return [Task(task_id="placeholder-1",
                      prompt="Frontier-Bench v0.1 wiring lands in iteration 2.",
                      reference="",
                      metadata={"scaffold": True})][:limit] if limit else [
            Task(task_id="placeholder-1",
                 prompt="Frontier-Bench v0.1 wiring lands in iteration 2.",
                 reference="", metadata={"scaffold": True})]

    def score(self, output, task):
        return Score(passed=False, details="scaffold: oracle not wired yet")


class Nl2RepoAdapter(SuiteAdapter):
    name = "nl2repo"
    category = "coding"
    description = "NL2Repo (natural-language to repository generation)"
    source = "NL2Repo official release (id TBD — verify)"
    status = "scaffold"

    def tasks(self, limit=None):
        return [Task(task_id="placeholder-1",
                      prompt="NL2Repo wiring lands in iteration 2.",
                      reference="", metadata={"scaffold": True})]

    def score(self, output, task):
        return Score(passed=False, details="scaffold: oracle not wired yet")


class SweAtlasQnaAdapter(SuiteAdapter):
    name = "swe-atlas-qna"
    category = "coding"
    description = "SWE Atlas – QnA (software-engineering question answering)"
    source = "SWE Atlas QnA official release (id TBD — verify)"
    status = "scaffold"

    def tasks(self, limit=None):
        return [Task(task_id="placeholder-1",
                      prompt="SWE Atlas QnA wiring lands in iteration 2.",
                      reference="", metadata={"scaffold": True})]

    def score(self, output, task):
        return Score(passed=False, details="scaffold: oracle not wired yet")


class _TerminalBenchBase(SuiteAdapter):
    category = "coding"
    status = "scaffold"

    def requirements(self):
        return [
            Requirement("pip", "terminal_bench", "official terminal-bench package"),
            Requirement("cli", "docker", "task containers + oracle"),
        ]

    def tasks(self, limit=None):
        try:
            from terminal_bench.dataset import load_datasets  # type: ignore
        except Exception:
            return [Task(task_id="missing-deps",
                         prompt="",
                         reference="",
                         metadata={"skip_reason": "terminal_bench package not installed"})]
        try:
            names = load_datasets()
        except Exception as exc:
            return [Task(task_id="load-error", prompt="", reference="",
                         metadata={"skip_reason": f"dataset load failed: {exc}"})]
        tasks = [Task(task_id=n, prompt=f"Terminal-Bench task: {n}",
                      reference="", metadata={"tb_task": n}) for n in names]
        return tasks[:limit] if limit else tasks

    def score(self, output, task):
        if task.metadata.get("skip_reason"):
            return Score(passed=False, details=task.metadata["skip_reason"])
        return Score(passed=False, details="scaffold: tb agent loop lands in iteration 2")


class TbTerminusAdapter(_TerminalBenchBase):
    name = "tb-terminus"
    description = "Terminal-Bench 2.1 via Terminus-2 harness"
    source = "terminal-bench 2.1 Terminus-2 harness"


class TbClaudeAdapter(_TerminalBenchBase):
    name = "tb-claude"
    description = "Terminal-Bench 2.1 via Claude Code harness"
    source = "terminal-bench 2.1 Claude Code harness"
