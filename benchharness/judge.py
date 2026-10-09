"""LLM-as-judge grading, ported from openai/simple-evals.

The grader template and `Correct: (yes|no)` verdict protocol are the
canonical ones used for HLE and BrowseComp. Disabled by default (it doubles
inference cost); enable with BENCH_JUDGE=1 / --judge for judge verdicts on
top of the heuristic scores.
"""

from __future__ import annotations

import re

GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available.
""".strip()


def parse_verdict(text: str) -> bool | None:
    """Extract `correct: yes|no` (last occurrence wins). None if absent."""
    matches = re.findall(r"(?im)^correct\s*:\s*\(?(yes|no)\)?\s*$", text)
    if not matches:
        matches = re.findall(r"(?i)correct\s*:\s*\(?(yes|no)\)?", text)
    if not matches:
        return None
    return matches[-1].lower() == "yes"


def judge_correct(client, model: str, question: str, correct_answer: str,
                  response: str, max_tokens: int = 1024) -> tuple[bool | None, str]:
    """Ask the judge model; returns (verdict, raw_judge_text)."""
    prompt = GRADER_TEMPLATE.format(question=question, correct_answer=correct_answer,
                                    response=response)
    resp = client.chat([{"role": "user", "content": prompt}], model=model,
                       max_tokens=max_tokens)
    raw = client.extract_text(resp)
    return parse_verdict(raw), raw
