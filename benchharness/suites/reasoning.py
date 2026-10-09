"""Reasoning suites: HLE (no tools / with tools) and GPQA Diamond.

These are fully runnable: HuggingFace datasets when the optional `datasets`
dependency is installed (else bundled smoke samples), deterministic scorers.
"""

from __future__ import annotations

import re

from benchharness.schema import Score
from benchharness.suites.base import Requirement, SuiteAdapter, Task, strip_thinking

GPQA_SYSTEM = (
    "You are answering a multiple-choice science question. "
    "Reply with ONLY the letter of the correct answer: A, B, C, or D."
)
HLE_SYSTEM = (
    "You are answering an expert-level exam question. Give the exact final "
    "answer with no explanation unless the question asks for working."
)


def _load_hf_dataset(name: str, config: str | None = None):
    """Import-gated HF loader; returns None when unavailable (offline/smoke).

    Tries test -> train -> validation splits (SWE-bench/HLE publish `test`;
    GPQA publishes `train`). Gated datasets (gpqa, hle) need `HF_TOKEN`.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:
        return None
    for split in ("test", "train", "validation", "full"):
        try:
            if config:
                return load_dataset(name, config, split=split)
            return load_dataset(name, split=split)
        except Exception:
            continue
    return None


def extract_letter(text: str) -> str:
    """First A-D letter, preferring explicit 'Answer: X' markers."""
    m = re.search(r"(?i)answer\s*[:\-]?\s*\(?([A-D])\)?", text)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-D])\b", text)
    return m.group(1).upper() if m else ""


def normalize_answer(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[.$,;:!?\"'`()\[\]{}]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class GpqaDiamondAdapter(SuiteAdapter):
    """GPQA Diamond: 198 expert multiple-choice science questions."""

    name = "gpqa-diamond"
    category = "reasoning"
    description = "GPQA Diamond (Idavidrein/gpqa) multiple-choice, exact-letter grading"
    source = "Idavidrein/gpqa :: gpqa_diamond"
    status = "wired"

    SMOKE = [
        ("smoke-1", "What is 2+2? A) 3 B) 4 C) 5 D) 6", "B"),
        ("smoke-2", "Which planet is known as the Red Planet? A) Venus B) Jupiter C) Mars D) Saturn", "C"),
    ]

    def requirements(self):
        return [Requirement("pip", "datasets", "HF dataset loader (else smoke samples)", soft=True)]

    def tasks(self, limit=None):
        rows = _load_hf_dataset("Idavidrein/gpqa", "gpqa_diamond")
        out: list[Task] = []
        if rows is not None:
            for i, row in enumerate(rows):
                choices = [row.get("Correct Answer", ""),
                           row.get("Incorrect Answer 1", ""),
                           row.get("Incorrect Answer 2", ""),
                           row.get("Incorrect Answer 3", "")]
                letters = "ABCD"
                prompt = row.get("Question", "") + "\n" + "\n".join(
                    f"{letters[j]}) {c}" for j, c in enumerate(choices)
                )
                out.append(Task(task_id=f"gpqa-{i}", prompt=prompt,
                                reference="A", system=GPQA_SYSTEM))
        else:
            for tid, prompt, ref in self.SMOKE:
                out.append(Task(task_id=tid, prompt=prompt, reference=ref,
                                system=GPQA_SYSTEM))
        return out[:limit] if limit else out

    def score(self, output, task):
        got = extract_letter(strip_thinking(output))
        ok = bool(got) and got == task.reference.upper()
        return Score(passed=ok, details=f"got={got!r} want={task.reference!r}")


class HleAdapter(SuiteAdapter):
    """HLE without tools: free-response expert questions, normalized match."""

    name = "hle"
    category = "reasoning"
    description = "Humanity's Last Exam (cais/hle), no tools, normalized-answer grading"
    source = "cais/hle"
    status = "wired"

    SMOKE = [
        ("smoke-1", "What is the capital of France?", "Paris"),
        ("smoke-2", "What is 7 * 8?", "56"),
    ]

    def requirements(self):
        return [Requirement("pip", "datasets", "HF dataset loader (else smoke samples)", soft=True)]

    def tasks(self, limit=None):
        rows = _load_hf_dataset("cais/hle")
        out: list[Task] = []
        if rows is not None:
            for row in rows:
                out.append(Task(task_id=str(row.get("id", len(out))),
                                prompt=str(row.get("question", "")),
                                reference=str(row.get("answer", "")),
                                system=HLE_SYSTEM))
        else:
            for tid, prompt, ref in self.SMOKE:
                out.append(Task(task_id=tid, prompt=prompt, reference=ref,
                                system=HLE_SYSTEM))
        return out[:limit] if limit else out

    def score(self, output, task):
        want = normalize_answer(task.reference)
        got = normalize_answer(strip_thinking(output))
        ok = bool(want) and (got == want or want in got)
        return Score(passed=ok, details=f"normalized_match={ok}")

    def score_with_client(self, output, task, client, model):
        from benchharness.judge import judge_correct

        verdict, _ = judge_correct(client, model, task.prompt, task.reference,
                                   strip_thinking(output))
        if verdict is None:
            return None
        return Score(passed=verdict, score=1.0 if verdict else 0.0,
                     details="canonical simple-evals grader")


class HleToolsAdapter(HleAdapter):
    """HLE with tools: same questions, model may call a sandboxed python tool."""

    name = "hle-tools"
    category = "reasoning"
    description = "HLE with a sandboxed python tool (multi-turn tool loop)"
    status = "wired"
    max_tokens = 4096

    TOOL_SYSTEM = (
        HLE_SYSTEM + " You have a tool: run_python(code) executes Python and "
        "returns stdout. To use it, reply with a single fenced block starting "
        "with ```tool then the code. You get at most 3 tool rounds, then give "
        "the final answer as plain text."
    )

    def tasks(self, limit=None):
        base = super().tasks(limit)
        for t in base:
            t.system = self.TOOL_SYSTEM
            t.metadata["tool_loop"] = "run_python"
            t.metadata["tool_rounds"] = 3
        return base
