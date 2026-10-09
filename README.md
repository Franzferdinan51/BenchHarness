# BenchHarness

Unified multi-suite benchmark harness for local LLMs. LM Studio-first: it pings
your local server, picks a chat model, and runs coding / reasoning / agentic
suites with one command.

LM Studio conventions (base URL, env vars, model picker) are ported from
[local-grok-cli](https://github.com/Franzferdinan51/local-grok-cli) so both
tools share one setup.

## Suites

| Suite | Category | Status |
|---|---|---|
| `tb-terminus` | coding | wired — TB 2.1 Harbor `--agent terminus-2`, LM Studio-routed (oracle PASS live) |
| `tb-claude` | coding | wired — TB 2.1 Harbor `--agent claude-code` (needs `ANTHROPIC_API_KEY`) |
| `tb-hermes` | coding | wired — TB 2.1 Harbor `--agent hermes` (Hermes-native model config) |
| `hermes-bench` | agentic | scaffold — Nous HermesIndex suite is closed; `tb-hermes` is the proxy |
| `swe-verified` | coding | wired — official docker FAIL_TO_PASS eval (gold-validated; heuristic fallback) |
| `swe-pro` | coding | wired — Pro V2 Harbor tasks (`harbor run -p v2/tasks`, oracle PASS live; amd64 pre-pull on ARM) |
| `swe-multilingual` | coding | wired — official docker eval when available, else heuristic |
| `deepswe` | coding | wired — EvalScope Pier agent on `evalscope/deep-swe` (113 tasks, verifier acc; path proven end-to-end, live LM trial deferred to GPU-free) |
| `frontier-bench` | coding | scaffold — Anthropic-reported, no public artifact (TB4 line tracked) |
| `nl2repo` | coding | wired — shell-agent loop in per-task image + `verify_cmd` grading |
| `swe-atlas-qna` | coding | wired — `ScaleAI/SWE-Atlas-QnA`, rubric-keyword recall (LLM judge next) |
| `hle` | reasoning | wired — `cais/hle`, normalized-answer grading |
| `hle-tools` | reasoning | wired — HLE + sandboxed `run_python` tool loop |
| `gpqa-diamond` | reasoning | wired — `Idavidrein/gpqa`, exact-letter grading |
| `mcp-atlas` | agentic | wired — fidelity ladder: real MCP tool loop + claim judge via agent-environment when up (else GTFA recall) |
| `toolathlon` | agentic | wired — official remote service, private mode (108 Verified tasks, acc; path proven end-to-end incl. WS relay; one job at a time on the public service) |
| `widesearch` | agentic | wired — `ByteDance-Seed/WideSearch`, column recall (cell judge next) |
| `browsecomp` | agentic | wired — decrypted `smolagents/browse_comp`, containment (LLM grader next) |
| `claweval` | agentic | wired — EvalScope pinned official runner (300 tasks, Pass³ via `BENCH_CLAW_TRIALS=3`; live trial deferred to GPU-free) |
| `demo` | reasoning | wired golden suite (2 tasks, no deps) |

`wired` suites run end-to-end today. Without the optional `datasets` package
(or offline), dataset suites fall back to bundled smoke samples so the full
pipeline stays exercisable. `scaffold` suites run through the runner and report
honestly instead of failing.

Gated HuggingFace datasets (`Idavidrein/gpqa`, `cais/hle`) need access plus
`HF_TOKEN` in the environment (or `huggingface-cli login`); without it those
suites use smoke samples or defer. SWE-bench Verified / Multilingual,
SWE-Atlas-QnA, WideSearch, MCP-Atlas, NL2Repo, and BrowseComp are open and
load directly. DeepSWE uses the ungated `evalscope/deep-swe` ModelScope
mirror; SWE-bench Pro V2 uses its Harbor task tree (no HF needed).

SWE-bench notes: with the `swe` extra installed and Docker running,
`swe-verified` / `swe-multilingual` grade via the official swebench
FAIL_TO_PASS/PASS_TO_PASS docker eval (gold patch resolves, garbage does
not — both verified live). Images are amd64-only; the harness pre-pulls
with `--platform linux/amd64` on Apple Silicon. Set `BENCH_SWE_DOCKER=0`
to force fast heuristic grading.

Terminal-Bench notes (validated live on Apple Silicon + Colima):

- `uv tool install harbor` plus a running Docker daemon are required.
- Give Colima room (`colima start --memory 24` on a 128GB host); the
  harness passes `--memory ignore` by default because TB's 2G task limits
  OOM-kill amd64 images under qemu emulation (override with
  `BENCH_HARBOR_MEMORY`).
- Keep `--out` under `$HOME`: Colima bind mounts don't propagate `/tmp`,
  so rewards never download from jobs run there (the harness warns).
- Default dataset is `terminal-bench/terminal-bench-2-1` (89 tasks,
  org/name registry form, oracle-validated live); legacy
  `terminal-bench@2.0` via `TB_DATASET` override.
- `BENCH_HARBOR_AGENT=oracle` runs golden solutions with zero LM calls —
  the recommended smoke test for the TB path.
- `tb-claude` needs `ANTHROPIC_API_KEY` (Claude Code CLI); `tb-hermes`
  uses your Hermes CLI provider config for model routing.

EvalScope notes (`claweval`, `deepswe` — validated infra, live agent
trials deferred until the GPU is free):

- These suites run through EvalScope's official runners in an isolated
  venv (heavy deps stay out of this project). Set it up once:
  `uv venv ~/.cache/benchharness/evalscope-venv && uv pip install
  --python ~/.cache/benchharness/evalscope-venv/bin/python
  'evalscope[deep_swe]' 'claw-eval[sandbox,mock,web] @
  git+https://github.com/claw-eval/claw-eval.git@d3f02d4'`
  (override the interpreter with `BENCH_EVALSCOPE_PYTHON`).
- Agent and LLM judge both point at LM Studio (`--api-url` /
  judge `api_url`); DeepSWE uses the litellm model class for
  OpenAI-compatible endpoints, with the model sent as
  `openai/<name>` plus `OPENAI_API_BASE`/`OPENAI_API_KEY` into the
  agent container (localhost rewritten to `host.docker.internal`;
  override with `BENCH_CONTAINER_HOST`).
- DeepSWE note: Pier's egress proxy only allows ports 80/443, so the
  driver flips `allow_internet` on the EvalScope-cached snapshot at
  run time (the store self-restores edits, hence the wrap) to give
  the agent direct egress to local endpoints. The verifier runs
  unchanged. Set `BENCH_DEEPSWE_ALLOW_INTERNET=0` for the locked
  protocol (local endpoints then unreachable).
- Keep `--out` under `$HOME` for EvalScope suites too: Colima bind
  mounts don't propagate `/tmp`, so reward files never download
  back (the harness warns).
- `BENCH_CLAW_SUBSET=general,multimodal,multi_turn` selects Claw-Eval
  splits (default `general`); `BENCH_CLAW_TRIALS=3` / `BENCH_DEEPSWE_TRIALS`
  set repeats (3 = official Pass³ — pass requires every trial to pass).
- The Claw-Eval sandbox image (`claw-eval-agent:latest`) builds once
  from the pinned official Dockerfile; fixtures (~3GB) download once
  from ModelScope. Both are cached after the first run.
- `toolathlon` submits to the official remote service (private mode):
  envs and scoring run remotely while model calls relay back to local
  LM Studio over a WebSocket proxy. The public service runs one job at
  a time — a 503 "Server is busy" means retry later (the harness
  reports the service message verbatim).
- `mcp-atlas` prefers agent mode when the official `agent-environment`
  service is up: `docker run -d -p 1984:1984
  ghcr.io/scaleapi/mcp-atlas:1.2.5` (13/20 servers live keyless; add
  API keys via env file for the rest). Override the URL with
  `BENCH_MCP_ENV_URL`. One EvalScope batch job runs the tool loop +
  per-claim judge; rows map back by exact prompt; tasks needing
  offline servers report `excluded` honestly. Without the service the
  suite falls back to GTFA claim-keyword recall.

Notes from live testing against local reasoning models (Ornith/Qwen3-style):

- Scorers strip `<think>...</think>` blocks before grading, so decoy letters
  and draft diffs in the trace can't pollute the verdict.
- Thinking models can spend thousands of tokens reasoning before answering;
  SWE suites default to a 16384 completion cap (`--max-tokens` overrides).
- The picker prefers the loaded model; set `LM_STUDIO_MODEL` to force one.
- LLM-judge grading (canonical simple-evals `correct: yes|no` protocol) is
  available for HLE, BrowseComp, and SWE-Atlas QnA via `--judge` /
  `BENCH_JUDGE=1`. Off by default (doubles inference); judge verdicts are
  recorded alongside the heuristic score in each result's details.
- Tasks run in parallel (`--jobs`, default 4) across all suites in one pool;
  each task has a 600s wall-clock budget (`per_task_timeout_secs`) so one
  hung generation can't stall a run. Note LM Studio itself may serialize
  requests against a single loaded model.

## Setup

```sh
git clone https://github.com/Franzferdinan51/BenchHarness.git
cd BenchHarness
uv venv
uv pip install -e ".[datasets]"   # datasets/dev extras as needed
```

Make sure LM Studio serves the OpenAI-compatible API (default
`http://127.0.0.1:1234/v1`; override with `GROK_MODELS_BASE_URL` or
`LM_STUDIO_URL`). Pick a model with `LM_STUDIO_MODEL`, or let the harness use
the first loaded chat model. If your server needs auth, set `LM_STUDIO_API_KEY`.

## Usage

```sh
uv run bench-harness doctor                    # ping LM Studio, resolve model
uv run bench-harness list-suites               # all 20 registered suites
uv run bench-harness run --suite demo          # golden smoke run
uv run bench-harness run --suite gpqa-diamond --limit 5
uv run bench-harness run --suite swe-verified --suite hle --jobs 4
uv run bench-harness run --suite all --limit 2 --out bench-results
uv run bench-harness run --suite swe-pro --task openlibrary-00bec1e7   # one task by id substring
BENCH_HARBOR_AGENT=oracle uv run bench-harness run --suite swe-pro --limit 1  # LM-free Harbor smoke
uv run bench-harness compare bench-results/<run-a> bench-results/<run-b>  # per-suite pass@1 + delta
uv run bench-harness inspect bench-results/<run-id>
uv run bench-harness export bench-results/<run-id> --format json
```

Results stream to `<out>/<run-id>/results.jsonl` (crash-safe, resumable with
`--resume`) plus an aggregate `summary.json` with pass@1 / mean scores.

## Development

```sh
uv run pytest            # unit + mock-LM + golden-suite tests
cp bench.toml.example bench.toml   # optional local config
```

Architecture: `benchharness/lm_client.py` (LM Studio client),
`suites/base.py` (`SuiteAdapter` contract), `suites/coding|reasoning|agentic.py`
(adapters), `runner.py` (parallel runner + tool loop), `sandbox.py`
(local/docker exec), `schema.py` (JSONL result records).
