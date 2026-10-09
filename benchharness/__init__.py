"""BenchHarness: unified multi-suite benchmark runner for local LLMs.

LM Studio conventions (base URL, env vars, model picker) are ported from
https://github.com/Franzferdinan51/local-grok-cli so both tools share one setup.
"""

from benchharness.schema import RunSummary, Score, TaskResult
from benchharness.config import BenchConfig
from benchharness.lm_client import LMStudioClient, DiscoveredModel
from benchharness.registry import REGISTRY, get_suite, list_suites

__all__ = [
    "BenchConfig",
    "DiscoveredModel",
    "LMStudioClient",
    "REGISTRY",
    "RunSummary",
    "Score",
    "TaskResult",
    "get_suite",
    "list_suites",
]

__version__ = "0.1.0"
