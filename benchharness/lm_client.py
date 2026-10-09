"""LM Studio client, ported from local-grok-cli's sampler/picker behavior.

- Discovery: native ``GET /api/v0/models`` first, fallback to OpenAI-compatible
  ``GET /v1/models``; embeddings hidden; loaded models sort first.
- Chat: OpenAI-compatible ``POST /chat/completions`` with ``stream=false``,
  ``Bearer`` auth (``LM_STUDIO_API_KEY`` or dummy ``lm-studio``), retries.
- ``ping()``/``doctor()`` back the ``bench-harness doctor`` command.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from benchharness.config import BenchConfig


@dataclass
class DiscoveredModel:
    id: str
    loaded: bool = True
    max_context_length: int | None = None
    kind: str = ""


def is_chat_model(kind: str, model_id: str) -> bool:
    """Port of local-grok-cli is_chat_model: drop embeddings, keep llm/vlm."""
    k = kind.strip().lower()
    low = model_id.lower()
    if k in ("embeddings", "embedding"):
        return False
    if "embed" in low:
        return False
    return k in ("", "llm", "vlm")


def parse_models_payload(body: dict) -> tuple[list[DiscoveredModel], str]:
    """Return (models, shape) where shape is 'v0' or 'v1'."""
    data = body.get("data")
    if not isinstance(data, list):
        return [], "v1"
    first = data[0] if data and isinstance(data[0], dict) else {}
    v0 = "state" in first
    out: list[DiscoveredModel] = []
    seen: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id", "")).strip()
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        kind = str(item.get("type", "") or "")
        loaded = (
            str(item.get("state", "")).lower() == "loaded" if v0 else True
        )
        ctx = item.get("max_context_length", item.get("maxContextLength"))
        out.append(
            DiscoveredModel(
                id=model_id,
                loaded=loaded,
                max_context_length=int(ctx) if isinstance(ctx, (int, float)) else None,
                kind=kind,
            )
        )
    return out, ("v0" if v0 else "v1")


def _error_detail(resp: httpx.Response) -> str:
    """Best-effort server error message (LM Studio puts it in error.message)."""
    try:
        body = resp.json()
        if isinstance(body, dict):
            err = body.get("error", body)
            if isinstance(err, dict):
                return str(err.get("message", resp.text))[:300]
            return str(err)[:300]
    except ValueError:
        pass
    return resp.text[:300]


def select_chat_models(models: list[DiscoveredModel]) -> list[DiscoveredModel]:
    """Chat models only, loaded first (mirrors select_listed_models)."""
    chat = [m for m in models if is_chat_model(m.kind, m.id)]
    chat.sort(key=lambda m: not m.loaded)
    return chat


class LMStudioClient:
    def __init__(self, config: BenchConfig, transport: httpx.BaseTransport | None = None):
        self.config = config
        self._client = httpx.Client(
            base_url=config.base_url,
            headers={"Authorization": f"Bearer {config.api_key}"},
            timeout=httpx.Timeout(config.request_timeout_secs,
                                  connect=config.connect_timeout_secs),
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "LMStudioClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- discovery -----------------------------------------------------
    def _get_json(self, url: str) -> dict | None:
        try:
            resp = self._client.get(url)
            resp.raise_for_status()
            body = resp.json()
            return body if isinstance(body, dict) else None
        except (httpx.HTTPError, ValueError):
            return None

    def list_models(self) -> list[DiscoveredModel]:
        """Native /api/v0/models, then OpenAI-compat /v1/models fallback."""
        root = self.config.server_root
        for url in (f"{root}/api/v0/models", f"{root}/v1/models"):
            body = self._get_json(url)
            if body:
                models, _ = parse_models_payload(body)
                if models:
                    return select_chat_models(models)
        # Last resort: relative /models against the configured base.
        body = self._get_json("/models")
        if body:
            models, _ = parse_models_payload(body)
            return select_chat_models(models)
        return []

    def resolve_model(self) -> str:
        """Model pick, mirroring local-grok-cli resolution.

        Explicit id (CLI > env > file) wins when it matches a discovered
        model — exact, then unique substring/alias. A stale or unknown
        id warns and falls back to the first loaded chat model instead
        of failing requests later; empty preference auto-picks.
        """
        import sys as _sys

        models = self.list_models()
        want = (self.config.model or "").strip()
        if not models:
            if want:
                return want  # undiscoverable; trust the explicit id
            raise RuntimeError(
                "No chat model found on LM Studio. Set LM_STUDIO_MODEL or load "
                "a model (hint: `lms server start`, then load a model)."
            )
        ids = [m.id for m in models]
        if not want:
            return ids[0]
        if want in ids:
            return want
        cands = [i for i in ids if want in i or i in want]
        if len(cands) == 1:
            print(f"warning: model {want!r} fuzzy-matched to {cands[0]!r}",
                  file=_sys.stderr)
            return cands[0]
        print(f"warning: model {want!r} not loaded; using {ids[0]!r} "
              f"(loaded: {', '.join(ids[:8])})", file=_sys.stderr)
        return ids[0]

    def ping(self) -> dict:
        """Lightweight liveness check used by `doctor`."""
        started = time.monotonic()
        models = self.list_models()
        latency_ms = int((time.monotonic() - started) * 1000)
        return {
            "ok": bool(models),
            "latency_ms": latency_ms,
            "models": [m.id for m in models[:20]],
            "model_count": len(models),
            "base_url": self.config.base_url,
        }

    # -- chat ----------------------------------------------------------
    def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 2048,
    ) -> dict:
        """POST /chat/completions with retries. Returns the raw response dict."""
        payload = {
            # runner/doctor validate once via resolve_model(); the hot path
            # trusts the configured id (no discovery call per request).
            "model": model or self.config.model or self.resolve_model(),
            "messages": messages,
            "temperature": self.config.temperature if temperature is None else temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                resp = self._client.post("/chat/completions", json=payload)
                if resp.status_code >= 400:
                    detail = _error_detail(resp)
                    if resp.status_code == 429 or resp.status_code >= 500:
                        last_error = RuntimeError(f"HTTP {resp.status_code}: {detail}")
                        time.sleep(min(2.0 * (attempt + 1), 8.0))
                        continue
                    raise RuntimeError(f"HTTP {resp.status_code}: {detail}")
                data = resp.json()
                if isinstance(data, dict):
                    return data
                last_error = ValueError("non-dict chat response")
            except httpx.HTTPError as exc:
                last_error = exc
                time.sleep(min(2.0 * (attempt + 1), 8.0))
        raise RuntimeError(f"chat completions failed after retries: {last_error}")

    @staticmethod
    def extract_text(response: dict) -> str:
        try:
            return response["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            return ""

    @staticmethod
    def extract_reasoning(response: dict) -> str:
        try:
            return response["choices"][0]["message"].get("reasoning_content") or ""
        except (KeyError, IndexError, TypeError, AttributeError):
            return ""

    @staticmethod
    def extract_finish_reason(response: dict) -> str:
        try:
            return response["choices"][0].get("finish_reason") or ""
        except (KeyError, IndexError, TypeError, AttributeError):
            return ""

    @staticmethod
    def extract_usage(response: dict) -> tuple[int, int]:
        usage = response.get("usage") or {}
        return int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))
