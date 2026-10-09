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
| `tb-terminus` | coding | scaffold — Terminus-2 harness loop lands next iteration |
| `tb-claude` | coding | scaffold — Claude Code harness loop lands next iteration |
| `swe-verified` | coding | wired — HF dataset + patch scoring (docker FAIL_TO_PASS eval next) |
| `swe-pro` | coding | wired — `ScaleAI/SWE-bench_Pro` (lowercase fail/pass keys mapped) |
| `swe-multilingual` | coding | wired — same loader/scorer as swe-verified |
| `deepswe` | coding | scaffold — official dataset id TBD |
| `frontier-bench` | coding | scaffold — official dataset id TBD |
| `nl2repo` | coding | scaffold — official dataset id TBD |
| `swe-atlas-qna` | coding | scaffold — official dataset id TBD |
| `hle` | reasoning | wired — `cais/hle`, normalized-answer grading |
| `hle-tools` | reasoning | wired — HLE + sandboxed `run_python` tool loop |
| `gpqa-diamond` | reasoning | wired — `Idavidrein/gpqa`, exact-letter grading |
| `mcp-atlas` | agentic | scaffold — tool loop next iteration |
| `toolathlon` | agentic | scaffold — tool loop next iteration |
| `widesearch` | agentic | scaffold — tool loop next iteration |
| `browsecomp` | agentic | scaffold — tool loop next iteration |
| `claweval` | agentic | scaffold — tool loop next iteration |
| `demo` | reasoning | wired golden suite (2 tasks, no deps) |

`wired` suites run end-to-end today. Without the optional `datasets` package
(or offline), dataset suites fall back to bundled smoke samples so the full
pipeline stays exercisable. `scaffold` suites run through the runner and report
honestly instead of failing.

Gated HuggingFace datasets (`Idavidrein/gpqa`, `cais/hle`) need access plus
`HF_TOKEN` in the environment (or `huggingface-cli login`); without it those
suites use smoke samples. SWE-bench Verified / Multilingual / Pro
(`ScaleAI/SWE-bench_Pro`) are open and load directly.

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
uv run bench-harness list-suites               # all 18 registered suites
uv run bench-harness run --suite demo          # golden smoke run
uv run bench-harness run --suite gpqa-diamond --limit 5
uv run bench-harness run --suite swe-verified --suite hle --jobs 4
uv run bench-harness run --suite all --limit 2 --out bench-results
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
