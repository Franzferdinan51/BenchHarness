# Live results log

Real-model trials against LM Studio (already-loaded models only; the
harness never loads/unloads). Each entry: date, suite/task, model,
verdict, tokens, notes.

## 2026-10-09 — tb-terminus / build-pmars — INFRA FAIL

- Model: `duckbot-ornith-1.5-35b-a3b-mlx@8bit` (LM Studio, already loaded)
- Command: `bench-harness run --suite tb-terminus --task build-pmars`
- Wall time: 630s (10.5 min). Tokens: 491,045 prompt / 40,679 completion.
- Verdict: infrastructure failure, not a model score. Colima's docker
  daemon died mid-run (`dial unix .../docker.sock: no such file`); the
  verifier failed at `AddTestsDirError` and no trial result was produced.
- Model behavior note: across 59 agent steps the model emitted malformed
  `<tool_call>` pseudo-XML instead of terminus-2's JSON schema, so the
  parser rejected every turn and zero commands executed. Open question
  whether a local-model-friendly system-prompt shim is needed for
  terminus-2, or a larger-context model run.
- Harness fix (Iteration 22): `require_docker_daemon()` preflight in
  TB/swe-pro `prepare()` fails fast when the daemon is down instead of
  burning GPU minutes; Colima restarted before the next trial.
- Next: re-run the same trial with a healthy daemon to get a real score.

## 2026-10-09 — tb-terminus / build-pmars — FAIL (reward 0.0)

- Model: `duckbot-ornith-1.5-35b-a3b-mlx@8bit` (LM Studio, already loaded)
- Command: `bench-harness run --suite tb-terminus --task build-pmars`
- Wall time: 220s (3.7 min). Tokens: 44,267 prompt / 5,575 completion.
- Verdict: genuine scored FAIL. 12 agent steps, only 1 parser error;
  the agent executed real commands, believed it had finished ("tested
  successfully outputting 'Results: 2 40 8'"), but the verifier returned
  reward 0.0. First real tb-terminus data point on this model.
- Notes: trials 4-6 were lost to session SIGTERM / Colima-down states
  (trial 6 proved the Iteration 22 preflight: fail-fast, zero GPU burn).
  Harbor removed its own container afterwards; /tmp empty, RAM recovered.
- Next: second task sample to see if build-pmars is representative.

## 2026-10-09 — tb-terminus / fix-git — FAIL (AgentTimeoutError)

- Model: `duckbot-ornith-1.5-35b-a3b-mlx@8bit` (LM Studio, already loaded)
- Command: `bench-harness run --suite tb-terminus --task fix-git --sequential`
- Wall time: 967s (16 min, agent-side timeout). Tokens: 92,641 / 21,413.
- Verdict: FAIL. 29 of 30 steps rejected by the terminus-2 parser — the
  model emitted `<tool_call>` pseudo-XML instead of the JSON schema, so
  ~zero commands executed before the agent timed out.
- Pattern: TB7 executed real commands cleanly (1 parse error) while TB3
  and TB8 collapsed into pseudo-XML. The model's terminus-2 schema
  adherence is inconsistent run-to-run — a format-repair shim or a
  stricter local-model system prompt is the likely next harness fix.
- Cleanup verified: Harbor removed its container, /tmp empty, 61% free.

## 2026-10-09 — swe-verified (0/2) + gpqa-diamond smoke (2/2)

- Model: `duckbot-ornith-1.5-35b-a3b-mlx@8bit`, sequential.
- swe-verified/astropy-12907 FAIL: model generated 32,767 tokens (hit the
  cap, finish=length) without emitting any patch — rambled instead.
- swe-verified/astropy-13033 FAIL: model emitted a real diff, but it was
  truncated mid-line, so the official docker eval failed at patch-apply
  (`patch unexpectedly ends in middle of line`). Genuine model FAIL;
  the harness docker-eval path itself is proven (container ran, report
  parsed, `error_ids` surfaced correctly).
- gpqa-diamond smoke-1/smoke-2 PASS (1.1s / 0.6s). Real GPQA needs
  HF_TOKEN (gated); smoke fallback works as designed.
- Note: an earlier identical run errored 4x with 600s timeouts because
  LM Studio was wedged right after the 16-min TB8 trial; server was
  healthy minutes later (demo 2/2). Lesson: pause briefly after long
  agent runs before chat batches.

## 2026-10-09 — hle / widesearch / browsecomp / swe-atlas-qna (1/4)

- Model: `duckbot-ornith-1.5-35b-a3b-mlx@8bit`, sequential, limit 1 each.
- hle/smoke-1 PASS (0.6s). Real HLE gated (needs HF_TOKEN); smoke ok.
- widesearch/ws_en_001 scored 2/8 column recall (20s, real WideSearch row).
- browsecomp/browsecomp-0 containment=False (4.3s, real decrypted row).
- swe-atlas-qna real rubric task: recall 8/99 (2.5s).
- All genuine scored results; no errors, no infra issues.
