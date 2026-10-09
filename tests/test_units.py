"""Unit tests: schema roundtrip, config precedence, picker, scorers."""

from __future__ import annotations

import json

from benchharness.config import BenchConfig
from benchharness.lm_client import (
    is_chat_model,
    parse_models_payload,
    select_chat_models,
)
from benchharness.schema import RunSummary, TaskResult
from benchharness.suites.coding import extract_patch, touched_files
from benchharness.suites.reasoning import extract_letter, normalize_answer


def test_task_result_jsonl_roundtrip():
    r = TaskResult(run_id="r1", model="m", suite="demo", task_id="t1",
                   passed=True, score=1.0, latency_ms=12)
    assert TaskResult.from_json(r.to_json()) == r


def test_summary_aggregation():
    mk = lambda tid, passed, status="done", score=None: TaskResult(
        run_id="r", model="m", suite="demo", task_id=tid, passed=passed,
        score=(1.0 if passed else 0.0) if score is None else score, status=status)
    results = [mk("a", True), mk("b", False), mk("c", False, status="error"),
               mk("d", False, status="skipped")]
    s = RunSummary.from_results("r", "m", ["demo"], "t0", results)
    assert (s.total, s.passed, s.errors, s.skipped) == (4, 1, 1, 1)
    assert s.pass_at_1 == 1 / 3
    assert s.per_suite["demo"]["total"] == 4


def test_config_env_precedence(monkeypatch):
    monkeypatch.setenv("GROK_MODELS_BASE_URL", "http://x:1/v1")
    monkeypatch.setenv("LM_STUDIO_URL", "http://y:2/v1")
    monkeypatch.setenv("LM_STUDIO_MODEL", "model-a")
    monkeypatch.setenv("GROK_MODEL", "model-b")
    cfg = BenchConfig.load()
    assert cfg.base_url == "http://x:1/v1"  # GROK_MODELS_BASE_URL wins
    assert cfg.model == "model-a"  # LM_STUDIO_MODEL wins
    assert cfg.api_key == "lm-studio"  # dummy default


def test_config_chat_completions_suffix_stripped(monkeypatch):
    monkeypatch.setenv("LM_STUDIO_URL", "http://x:1/v1/chat/completions")
    assert BenchConfig.load().base_url == "http://x:1/v1"


def test_picker_hides_embeddings_and_prefers_v0(tmp_path):
    body = {"data": [
        {"id": "emb", "type": "embeddings", "state": "loaded"},
        {"id": "lib-model", "type": "llm", "state": "not-loaded"},
        {"id": "loaded-model", "type": "llm", "state": "loaded"},
        {"id": "sneaky-embed-llm", "type": "llm", "state": "loaded"},
    ]}
    models, shape = parse_models_payload(body)
    assert shape == "v0"
    chat = select_chat_models(models)
    assert [m.id for m in chat] == ["loaded-model", "lib-model"]
    assert not is_chat_model("embeddings", "x")
    assert not is_chat_model("llm", "nomic-embed-text")
    assert is_chat_model("", "whatever")


def test_v1_shape_marks_loaded():
    models, shape = parse_models_payload({"data": [{"id": "a", "object": "model"}]})
    assert shape == "v1" and models[0].loaded


def test_gpqa_letter_extraction():
    assert extract_letter("Answer: C") == "C"
    assert extract_letter("I think (B) is right") == "B"
    assert extract_letter("no letters here....".upper().lower()) in ("", "A", "B", "C", "D")


def test_normalize_answer():
    assert normalize_answer("  Paris. ") == "paris"
    assert normalize_answer("56") == "56"


def test_hf_loader_split_fallback(monkeypatch):
    """Loader tries test -> train -> validation before giving up."""
    import sys
    import types
    import benchharness.suites.reasoning as reasoning_mod

    calls: list[str] = []

    def fake_load_dataset(name, *args, **kwargs):
        calls.append(kwargs.get("split", ""))
        if kwargs.get("split") == "train":
            return ["row"]
        raise ValueError("no such split")

    stub = types.ModuleType("datasets")
    stub.load_dataset = fake_load_dataset
    monkeypatch.setitem(sys.modules, "datasets", stub)
    assert reasoning_mod._load_hf_dataset("x/y") == ["row"]
    assert calls == ["test", "train"]

    def always_fail(name, *args, **kwargs):
        raise ValueError("offline")

    stub.load_dataset = always_fail
    assert reasoning_mod._load_hf_dataset("x/y") is None


def test_swe_pro_lowercase_keys_mapped(monkeypatch):
    import benchharness.suites.coding as coding_mod
    from benchharness.registry import get_suite

    rows = [{"instance_id": "i1", "repo": "r", "base_commit": "c",
             "patch": "p", "problem_statement": "ps", "hints_text": "",
             "fail_to_pass": ["t1"], "pass_to_pass": ["t2"]}]
    monkeypatch.setattr(coding_mod, "_load_hf_dataset", lambda *a, **k: rows)
    tasks = get_suite("swe-pro").tasks()
    assert tasks[0].metadata["fail_to_pass"] == "['t1']"
    assert tasks[0].metadata["pass_to_pass"] == "['t2']"


def test_patch_extraction_and_files():
    text = '```diff\ndiff --git a/x.py b/x.py\n- a\n+ b\n```'
    patch = extract_patch(text)
    assert "diff --git" in patch
    assert touched_files(patch) == {"x.py"}
    assert extract_patch("no patch here") == ""
    # truncated output: unterminated fence still yields the patch
    assert extract_patch("```diff\ndiff --git a/x.py b/x.py\n- a") == \
        "diff --git a/x.py b/x.py\n- a"
    # non-diff fence without patch markers is not a patch
    assert extract_patch("```python\nprint(1)\n```") == ""
