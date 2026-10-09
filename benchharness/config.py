"""Configuration. Precedence mirrors local-grok-cli's endpoints module:

- Inference base URL: ``GROK_MODELS_BASE_URL`` > ``LM_STUDIO_URL`` > ``http://127.0.0.1:1234/v1``
- Bearer [REDACTED] ``LM_STUDIO_API_KEY`` else dummy ``lm-studio`` (unprotected servers)
- Model: ``LM_STUDIO_MODEL`` > ``GROK_MODEL`` > first loaded chat model
- Connect timeout: ``GROK_CONNECT_TIMEOUT_SECS`` (default 2 for pings)

A ``bench.toml`` file may set the same keys under ``[lm]``; env wins.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover (py<3.11 not supported anyway)
    import tomli as tomllib  # type: ignore[no-redef]

LM_STUDIO_BASE_URL_DEFAULT = "http://127.0.0.1:1234/v1"
LM_STUDIO_DUMMY_API_KEY = "lm-studio"
LM_STUDIO_API_KEY_ENV_VAR = "LM_STUDIO_API_KEY"

CONFIG_LOCATIONS = (
    Path("bench.toml"),
    Path.home() / ".grok-local" / "bench.toml",
    Path.home() / ".config" / "benchharness" / "bench.toml",
)


def _nonblank(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    return value.strip()


def _read_toml_file(path: Path) -> dict:
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return data


@dataclass
class BenchConfig:
    base_url: str = LM_STUDIO_BASE_URL_DEFAULT
    api_key: str = LM_STUDIO_DUMMY_API_KEY
    model: str | None = None
    temperature: float = 0.0
    connect_timeout_secs: float = 2.0
    request_timeout_secs: float = 300.0
    max_retries: int = 3
    jobs: int = 4
    max_tokens_override: int | None = None
    per_task_timeout_secs: float = 600.0
    out_dir: Path = field(default_factory=lambda: Path("bench-results"))
    source_file: str | None = None

    @classmethod
    def load(cls, cli_overrides: dict | None = None) -> "BenchConfig":
        file_data: dict = {}
        source: str | None = None
        for loc in CONFIG_LOCATIONS:
            if loc.is_file():
                file_data = _read_toml_file(loc)
                source = str(loc)
                break
        lm = file_data.get("lm", {}) if isinstance(file_data.get("lm"), dict) else {}
        run = file_data.get("run", {}) if isinstance(file_data.get("run"), dict) else {}
        cfg = cls(source_file=source)

        base = (
            _nonblank(os.environ.get("GROK_MODELS_BASE_URL"))
            or _nonblank(os.environ.get("LM_STUDIO_URL"))
            or _nonblank(lm.get("base_url"))
            or LM_STUDIO_BASE_URL_DEFAULT
        )
        cfg.base_url = base.rstrip("/")
        if cfg.base_url.endswith("/chat/completions"):
            cfg.base_url = cfg.base_url[: -len("/chat/completions")]

        cfg.api_key = (
            _nonblank(os.environ.get(LM_STUDIO_API_KEY_ENV_VAR))
            or _nonblank(lm.get("api_key"))
            or LM_STUDIO_DUMMY_API_KEY
        )
        cfg.model = (
            _nonblank(os.environ.get("LM_STUDIO_MODEL"))
            or _nonblank(os.environ.get("GROK_MODEL"))
            or _nonblank(lm.get("model"))
        )
        try:
            cfg.connect_timeout_secs = float(
                os.environ.get("GROK_CONNECT_TIMEOUT_SECS", lm.get("connect_timeout_secs", 2.0))
            )
        except ValueError:
            cfg.connect_timeout_secs = 2.0
        cfg.request_timeout_secs = float(lm.get("request_timeout_secs", 300.0))
        cfg.temperature = float(lm.get("temperature", 0.0))
        cfg.max_retries = int(lm.get("max_retries", 3))
        cfg.jobs = int(run.get("jobs", 4))
        mt = _nonblank(os.environ.get("BENCH_MAX_TOKENS")) or run.get("max_tokens")
        cfg.max_tokens_override = int(mt) if mt is not None else None
        cfg.per_task_timeout_secs = float(run.get("per_task_timeout_secs", 600.0))
        out = run.get("out_dir", "bench-results")
        cfg.out_dir = Path(out)

        for key, value in (cli_overrides or {}).items():
            if value is not None and hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg

    @property
    def server_root(self) -> str:
        """Origin without the trailing /v1, for native /api/v0/* calls."""
        base = self.base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        return base
