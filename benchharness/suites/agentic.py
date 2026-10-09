"""Agentic suites.

Wired with documented heuristic graders (parametric answers, no live tools):
- widesearch: ByteDance-Seed/WideSearch, required-column recall on the
  returned markdown table (per-cell llm_judge lands in iteration 4).
- browsecomp: smolagents/browse_comp mirror, canonical simple-evals decrypt,
  normalized-answer containment (LLM grader template lands in iteration 4).
- mcp-atlas: ScaleAI/MCP-Atlas, GTFA claim-keyword recall (MCP-server
  tool-call fidelity grading lands with the sandbox in iteration 4).

Deferred without burning model calls: toolathlon (needs 32 app
environments), claweval (source candidates unresolved).
"""

from __future__ import annotations

import base64
import hashlib
import json

from benchharness.schema import Score
from benchharness.suites.base import Requirement, SuiteAdapter, Task, strip_thinking
from benchharness.suites.coding import _keywords
from benchharness.suites.reasoning import _load_hf_dataset, normalize_answer


def derive_key(password: str, length: int) -> bytes:
    """Port of openai/simple-evals derive_key (SHA256, repeated)."""
    digest = hashlib.sha256(password.encode()).digest()
    return digest * (length // len(digest)) + digest[: length % len(digest)]


def browsecomp_decrypt(ciphertext_b64: str, password: str) -> str:
    """Port of openai/simple-evals decrypt (base64 + XOR)."""
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    return bytes(a ^ b for a, b in zip(encrypted, key)).decode()


WS_SYSTEM = (
    "Answer the research question by returning a single markdown table with "
    "one row per item found and one column per requested attribute. "
    "No explanation outside the table."
)


class WideSearchAdapter(SuiteAdapter):
    name = "widesearch"
    category = "agentic"
    description = "WideSearch (ByteDance-Seed/WideSearch, column-recall grade)"
    source = "ByteDance-Seed/WideSearch (200 tasks, MIT)"
    status = "wired"
    max_tokens = 8192

    SMOKE = [("smoke-en",
              "List the 3 largest oceans and their approximate area in km².",
              '["ocean", "area"]')]

    def requirements(self):
        return [Requirement("pip", "datasets", "HF loader (else smoke)", soft=True)]

    def tasks(self, limit=None):
        rows = _load_hf_dataset("ByteDance-Seed/WideSearch")
        out: list[Task] = []
        if rows is not None:
            for r in rows:
                out.append(Task(
                    task_id=str(r.get("instance_id", len(out))),
                    prompt=str(r.get("query", "")),
                    reference=str(r.get("evaluation", "")),
                    metadata={"language": r.get("language", "")},
                    system=WS_SYSTEM,
                ))
        else:
            for tid, prompt, ref in self.SMOKE:
                out.append(Task(task_id=tid, prompt=prompt, reference=ref,
                                system=WS_SYSTEM))
        return out[:limit] if limit else out

    @staticmethod
    def required_columns(evaluation: str) -> list[str]:
        try:
            data = json.loads(evaluation)
        except (ValueError, TypeError):
            return []
        if isinstance(data, list):  # smoke shape
            return [str(c) for c in data]
        if isinstance(data, dict):
            req = data.get("required", [])
            return [str(c) for c in req] if isinstance(req, list) else []
        return []

    def score(self, output, task):
        cols = self.required_columns(task.reference)
        if not cols:
            return Score(passed=False, details="no required columns parsed")
        table = strip_thinking(output).lower()
        hits = [c for c in cols if c.lower() in table]
        recall = len(hits) / len(cols)
        return Score(passed=recall >= 0.8, score=recall,
                     details=f"column_recall={len(hits)}/{len(cols)}; "
                             "per-cell llm_judge lands in iteration 4")


BC_QUERY_TEMPLATE = (
    "{Question}\n\nYour response should be in the following format:\n"
    "Explanation: {{your explanation for your final answer}}\n"
    "Exact Answer: {{your succinct, final answer}}\n"
    "Confidence: {{your confidence score between 0% and 100%}}"
)


class BrowseCompAdapter(SuiteAdapter):
    name = "browsecomp"
    category = "agentic"
    description = "BrowseComp (smolagents/browse_comp, decrypted, containment grade)"
    source = "smolagents/browse_comp mirror of openai/simple-evals (1,266 Qs)"
    status = "wired"
    max_tokens = 4096

    def requirements(self):
        return [Requirement("pip", "datasets", "HF loader (else smoke)", soft=True)]

    def tasks(self, limit=None):
        rows = _load_hf_dataset("smolagents/browse_comp")
        out: list[Task] = []
        if rows is not None:
            for i, r in enumerate(rows):
                try:
                    problem = browsecomp_decrypt(str(r.get("problem", "")),
                                                 str(r.get("canary", "")))
                    answer = browsecomp_decrypt(str(r.get("answer", "")),
                                                str(r.get("canary", "")))
                except Exception:
                    continue
                out.append(Task(
                    task_id=f"browsecomp-{i}",
                    prompt=BC_QUERY_TEMPLATE.format(Question=problem),
                    reference=answer,
                    metadata={"topic": str(r.get("problem_topic", ""))},
                ))
        else:
            out.append(Task(task_id="smoke-1",
                            prompt=BC_QUERY_TEMPLATE.format(
                                Question="What is the capital of France?"),
                            reference="Paris"))
        return out[:limit] if limit else out

    def score(self, output, task):
        want = normalize_answer(task.reference)
        got = normalize_answer(strip_thinking(output))
        ok = bool(want) and (got == want or want in got)
        return Score(passed=ok, details=f"containment={ok} (BENCH_JUDGE=1 for LLM grader)")

    def score_with_client(self, output, task, client, model):
        from benchharness.judge import judge_correct

        verdict, _ = judge_correct(client, model, task.prompt, task.reference,
                                   strip_thinking(output))
        if verdict is None:
            return None
        return Score(passed=verdict, score=1.0 if verdict else 0.0,
                     details="canonical simple-evals grader")


class McpAtlasAdapter(SuiteAdapter):
    name = "mcp-atlas"
    category = "agentic"
    description = "MCP-Atlas (ScaleAI/MCP-Atlas, GTFA claim recall)"
    source = "ScaleAI/MCP-Atlas (500 prompts, 36 MCP servers, 220 tools)"
    status = "wired"
    max_tokens = 4096

    SMOKE = [("smoke-1", "What year was the AssaultCube repo created?",
              "The AssaultCube GitHub repository was created in 2013.")]

    def requirements(self):
        return [Requirement("pip", "datasets", "HF loader (else smoke)", soft=True)]

    def tasks(self, limit=None):
        rows = _load_hf_dataset("ScaleAI/MCP-Atlas")
        out: list[Task] = []
        if rows is not None:
            for r in rows:
                tools = r.get("ENABLED_TOOLS", [])
                out.append(Task(
                    task_id=str(r.get("TASK", len(out))),
                    prompt=f"{r.get('PROMPT', '')}\n\n"
                           f"(Available tools in the full sandbox: "
                           f"{', '.join(tools[:12])}{'...' if len(tools) > 12 else ''})",
                    reference=str(r.get("GTFA_CLAIMS", "")),
                    metadata={"tools": list(tools) if isinstance(tools, list) else []},
                ))
        else:
            for tid, prompt, ref in self.SMOKE:
                out.append(Task(task_id=tid, prompt=prompt, reference=ref))
        return out[:limit] if limit else out

    def score(self, output, task):
        claims = str(task.reference or "")
        keys = _keywords(claims)
        if not keys:
            return Score(passed=False, details="no claim keywords")
        hits = {k for k in keys if k in strip_thinking(output).lower()}
        recall = len(hits) / len(keys)
        return Score(passed=recall >= 0.6, score=recall,
                     details=f"claim_recall={len(hits)}/{len(keys)}; "
                             "MCP tool-call fidelity grading lands in iteration 4")


class ToolathlonAdapter(SuiteAdapter):
    """Toolathlon / Tool Decathlon (hkust-nlp/Toolathlon, ICLR 2026).

    108 tasks across 32 apps / 604 tools with execution-based grading in
    realistic envs; trajectories dataset is gated. Deferred until the
    app-environment driver lands.
    """

    name = "toolathlon"
    category = "agentic"
    description = "Toolathlon-Verified (hkust-nlp/Toolathlon, env driver pending)"
    source = "github.com/hkust-nlp/Toolathlon + toolathlon.xyz"
    status = "scaffold"

    def requirements(self):
        return [Requirement("cli", "docker", "app environments")]

    def tasks(self, limit=None):
        return [Task(task_id="env-driver-pending", prompt="", reference="",
                     metadata={"skip_reason": "Toolathlon app-env driver lands "
                               "in iteration 4 (hkust-nlp/Toolathlon)"})]

    def score(self, output, task):
        return Score(passed=False, details="scaffold: env driver pending")


class HermesBenchAdapter(SuiteAdapter):
    """Hermes Bench (Nous HermesIndex suite: 150 tasks, 25 categories).

    As of iteration 6 the official suite has no public runner or task
    download — it runs inside Hermes Agent via the Nous portal
    (https://portal.nousresearch.com/bench). Runnable proxies in this
    harness: tb-hermes (same Hermes Agent harness on Terminal-Bench) and
    third-party am423/hermes-bench-tool-call (61 local tool-call tasks).
    """

    name = "hermes-bench"
    category = "agentic"
    description = "Hermes Bench (Nous HermesIndex; no public runner yet)"
    source = "https://portal.nousresearch.com/bench (closed suite)"
    status = "scaffold"

    def tasks(self, limit=None):
        return [Task(task_id="no-public-runner", prompt="", reference="",
                     metadata={"skip_reason": "Hermes Bench has no public "
                               "runner; use tb-hermes proxy or rerun research"})]

    def score(self, output, task):
        return Score(passed=False, details="scaffold: no public runner")


def _load_claweval_rows():
    """Raw (split, row) pairs from the ModelScope snapshot (bypasses the SDK
    builder, which is incompatible with datasets>=3). None when offline."""
    import os as _os

    wanted = [s.strip() for s in
              _os.environ.get("BENCH_CLAW_SUBSET", "general").split(",")
              if s.strip()] or ["general"]
    try:
        from modelscope import snapshot_download  # type: ignore
    except Exception:
        return None
    try:
        root = snapshot_download("claw-eval/Claw-Eval", repo_type="dataset")
        from datasets import load_dataset  # type: ignore
        out = []
        for split in wanted:
            ds = load_dataset("parquet",
                              data_files=f"{root}/data/{split}-00000-of-00001.parquet",
                              split="train")
            out.extend((split, dict(r)) for r in ds)
        return out
    except Exception:
        return None


class ClawEvalAdapter(SuiteAdapter):
    """Claw-Eval (claw-eval/Claw-Eval on ModelScope): 300 human-verified
    personal-assistant tasks (161 general / 101 multimodal / 38 multi_turn).

    Runs the pinned official runner + Docker sandbox + graders through
    EvalScope (evalscope_driver), with agent and LLM judge routed to
    LM Studio. BENCH_CLAW_SUBSET selects splits (default general);
    BENCH_CLAW_TRIALS sets repeats (3 = official Pass³)."""

    name = "claweval"
    category = "agentic"
    description = "Claw-Eval (EvalScope official runner, 300 tasks, Pass³)"
    source = "claw-eval/Claw-Eval on ModelScope via EvalScope claw_eval"
    status = "wired"

    def requirements(self):
        return [Requirement("pip", "modelscope", "ModelScope snapshot", soft=True),
                Requirement("pip", "datasets", "parquet reader", soft=True),
                Requirement("evalscope", "claw_eval",
                            "isolated EvalScope venv + pinned claw-eval"),
                Requirement("cli", "docker", "fixture sandboxes")]

    def tasks(self, limit=None):
        rows = _load_claweval_rows()
        if rows is None:
            return [Task(task_id="missing-data", prompt="", reference="",
                         metadata={"skip_reason": "Claw-Eval snapshot unavailable; "
                                   "pip install modelscope + network"})]
        out = [Task(
            task_id=str(r.get("task_id", i)),
            prompt=str(r.get("query", "")),
            reference="",
            metadata={"category": str(r.get("category", "")),
                      "language": str(r.get("language", "")),
                      "fixture": str(r.get("fixture", "")),
                      "split": split},
        ) for i, (split, r) in enumerate(rows)]
        return out[:limit] if limit else out

    def score(self, output, task):
        return Score(passed=False, details="claweval EvalScope trial (see run_external)")

    def run_external(self, task, ctx):
        import os as _os

        from benchharness.evalscope_driver import run_one

        config = ctx.get("config")
        model = str(ctx.get("model", ""))
        trials = int(_os.environ.get("BENCH_CLAW_TRIALS", "1"))
        timeout = float(getattr(config, "evalscope_timeout_secs", 3600.0))
        oc = run_one(
            "claw_eval", task.task_id, model=model,
            api_base=getattr(config, "base_url", "http://127.0.0.1:1234/v1"),
            api_key=getattr(config, "api_key", "lm-studio"),
            split=task.metadata.get("split", "general"),
            trials=trials, workdir=ctx["workdir"], timeout_secs=timeout,
        )
        output = f"claw_eval trials={trials} trace={oc.trace_path}"
        if oc.error:
            return output, Score(passed=False, score=0.0,
                                 details=f"{oc.details} [{oc.error}]")
        return output, Score(passed=oc.passed, score=oc.score, details=oc.details)
