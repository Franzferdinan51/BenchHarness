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

## 2026-10-09 — nl2repo / schema — FAIL (verify exit 127, capped at 20 turns)

- Model: `duckbot-ornith-1.5-35b-a3b-mlx@8bit`, sequential, 35.8s loop.
- Verdict: FAIL. Agent ran all 20 turns but the repo never reached a
  verifiable state (exit 127). Late turns degenerated (e.g. a markdown
  list executed as a shell command).
- Harness fix (Iteration 27): run_external now persists
  `transcript-<task>.json` (turns/commands/tokens) to the workdir and
  includes recent commands in the result excerpt; Score carries
  prompt/completion tokens through to TaskResult (was 0/0).

## 2026-10-09 — mcp-atlas — PASS (claim recall 10/16, score 0.625)

- Model: `duckbot-ornith-1.5-35b-a3b-mlx@8bit`, sequential, 12.2s.
- Recall mode (agent-environment server unreachable, as expected without
  the MCP env stack). First live PASS on a real agentic task.

## 2026-10-09 — claweval / T001zh_email_triage — PASS (score 1.0)

- Model: `duckbot-ornith-1.5-35b-a3b-mlx@8bit`, sequential, 42.9s.
- Two harness bugs found and fixed by this trial (Iteration 28):
  1. All three EvalScope callers passed `workdir=` to `run_one()`, whose
     kwarg is `work_dir` — every live claw/deepswe/toolathlon trial
     raised TypeError while mocks stayed green. Fixed + regression test
     (`test_run_one_kwarg_contract`, signature-bound strict stub).
  2. The official Dockerfile.agent defaults to the DaoCloud registry
     mirror, which 500s outside China. Harness now pre-builds
     `claw-eval-agent:latest` with `REGISTRY=docker.io` (the
     Dockerfile's own documented override) in `prepare()`; EvalScope
     reuses the image and skips its build.

## 2026-10-09 — deepswe / abs-module-cache-flags — FAIL (score 0.0)

- Model: `duckbot-ornith-1.5-35b-a3b-mlx@8bit`, sequential, 1934s (32 min).
- Genuine scored FAIL via the EvalScope Pier agent + verifier (acc 0.0).
  Longest live trial so far; no infra issues.

## 2026-10-09 — toolathlon / ab-testing — PASS (score 1.0)

- Model: `duckbot-ornith-1.5-35b-a3b-mlx@8bit`, 404s, via the official
  remote service + local relay. Genuine scored PASS.

## 2026-10-09 — hermes-bench / t01_echo — PASS (score 1.0)

- Model: `duckbot-ornith-1.5-35b-a3b-mlx@8bit`, 32s, real Hermes Agent
  routed to LM Studio (127.0.0.1:1234 verified in trace).
- Iteration 29 wired the suite: new `hermes_driver.py` (checkout mgmt,
  task enumeration, run_real subprocess, summary parsing) +
  HermesBenchAdapter (19/20 suites wired; only frontier-bench still a
  scaffold — no public artifact found on re-check).

## 2026-10-09 — final parallel run: hle-tools + hermes-bench (2/2 PASS)

- `--jobs 3` across suites: hle-tools/smoke-1 PASS (0.7s) and
  hermes-bench/t02_ls PASS (31s) in one parallel run. Parallel
  requests proven live end-to-end (plus `test_tasks_run_in_parallel`).
- Cleanup verified: zero leftover containers, /tmp empty, 92% mem free.
  ~8.4GB of dormant task images kept (re-pulls waste time, use no RAM).
