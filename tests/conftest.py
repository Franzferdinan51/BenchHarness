"""Shared fixtures: a mock LM Studio server over httpx.MockTransport."""

from __future__ import annotations

import json

import httpx
import pytest

from benchharness.config import BenchConfig


def make_transport(
    chat_text: str = "PINEAPPLE",
    models: list | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> httpx.MockTransport:
    if models is None:
        models = [
            {"id": "test-llm", "object": "model", "type": "llm", "state": "loaded"},
            {"id": "test-embed", "object": "model", "type": "embeddings", "state": "loaded"},
        ]

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/api/v0/models") or path.endswith("/v1/models") or path.endswith("/models"):
            return httpx.Response(200, json={"data": models, "object": "list"})
        if path.endswith("/chat/completions"):
            return httpx.Response(200, json={
                "choices": [{"message": {"role": "assistant", "content": chat_text}}],
                "usage": {"prompt_tokens": prompt_tokens,
                          "completion_tokens": completion_tokens},
            })
        return httpx.Response(404, json={"error": "not found"})

    return httpx.MockTransport(handler)


@pytest.fixture
def mock_config() -> BenchConfig:
    return BenchConfig(base_url="http://127.0.0.1:1234/v1", api_key="lm-studio",
                       model="test-llm", jobs=2)
