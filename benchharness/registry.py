"""Suite registry: all 17 goal suites plus the golden demo suite."""

from __future__ import annotations

from benchharness.suites.agentic import (
    BrowseCompAdapter,
    ClawEvalAdapter,
    HermesBenchAdapter,
    McpAtlasAdapter,
    ToolathlonAdapter,
    WideSearchAdapter,
)
from benchharness.suites.base import SuiteAdapter
from benchharness.suites.coding import (
    DeepSweAdapter,
    FrontierBenchAdapter,
    Nl2RepoAdapter,
    SweAtlasQnaAdapter,
    SweMultilingualAdapter,
    SweProAdapter,
    SweVerifiedAdapter,
    TbClaudeAdapter,
    TbHermesAdapter,
    TbTerminusAdapter,
)
from benchharness.suites.demo import DemoAdapter
from benchharness.suites.reasoning import (
    GpqaDiamondAdapter,
    HleAdapter,
    HleToolsAdapter,
)

REGISTRY: dict[str, type[SuiteAdapter]] = {}
for _cls in (
    TbTerminusAdapter,
    TbClaudeAdapter,
    TbHermesAdapter,
    SweVerifiedAdapter,
    SweProAdapter,
    SweMultilingualAdapter,
    DeepSweAdapter,
    FrontierBenchAdapter,
    Nl2RepoAdapter,
    SweAtlasQnaAdapter,
    GpqaDiamondAdapter,
    HleAdapter,
    HleToolsAdapter,
    McpAtlasAdapter,
    ToolathlonAdapter,
    WideSearchAdapter,
    BrowseCompAdapter,
    ClawEvalAdapter,
    HermesBenchAdapter,
    DemoAdapter,
):
    REGISTRY[_cls.name] = _cls


def get_suite(name: str) -> SuiteAdapter:
    try:
        return REGISTRY[name]()
    except KeyError:
        known = ", ".join(sorted(REGISTRY))
        raise KeyError(f"unknown suite {name!r}; known suites: {known}") from None


def list_suites(category: str | None = None) -> list[SuiteAdapter]:
    suites = [cls() for cls in REGISTRY.values()]
    if category:
        suites = [s for s in suites if s.category == category]
    return sorted(suites, key=lambda s: (s.category, s.name))
