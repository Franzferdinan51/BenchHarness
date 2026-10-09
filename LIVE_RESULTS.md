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
