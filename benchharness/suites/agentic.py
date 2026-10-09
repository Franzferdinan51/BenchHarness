"""Agentic suites: tool-use / browsing / MCP benchmarks.

Iteration-1 status is scaffold-with-contract: each adapter records its source,
declares the tools it will need, and exposes enumerable placeholder tasks so
the runner, registry, and reporting paths are exercised end-to-end. Full tool
loops land in iteration 2 (see tool_loop metadata consumed by runner).
"""

from __future__ import annotations

from benchharness.schema import Score
from benchharness.suites.base import SuiteAdapter, Task


class _AgenticScaffold(SuiteAdapter):
    category = "agentic"
    status = "scaffold"
    tools: tuple[str, ...] = ()

    def tasks(self, limit=None):
        return [Task(
            task_id="placeholder-1",
            prompt=f"{self.name} wiring lands in iteration 2.",
            reference="",
            metadata={"scaffold": True, "tools": list(self.tools)},
        )]

    def score(self, output, task):
        return Score(passed=False, details="scaffold: tool loop not wired yet")


class McpAtlasAdapter(_AgenticScaffold):
    name = "mcp-atlas"
    description = "MCP-Atlas (Model Context Protocol tool-use benchmark)"
    source = "MCP-Atlas official release (id TBD — verify)"
    tools = ("mcp_call",)


class ToolathlonAdapter(_AgenticScaffold):
    name = "toolathlon"
    description = "Toolathlon-Verified / Tool Decathlon (multi-tool agent tasks)"
    source = "Toolathlon-Verified official release (id TBD — verify)"
    tools = ("run_python", "web_fetch")


class WideSearchAdapter(_AgenticScaffold):
    name = "widesearch"
    description = "WideSearch (broad web-search research tasks)"
    source = "WideSearch official release (id TBD — verify)"
    tools = ("web_search", "web_fetch")


class BrowseCompAdapter(_AgenticScaffold):
    name = "browsecomp"
    description = "BrowseComp (hard browsing questions)"
    source = "BrowseComp official release (id TBD — verify)"
    tools = ("web_search", "web_fetch")


class ClawEvalAdapter(_AgenticScaffold):
    name = "claweval"
    description = "ClawEval (agent evaluation suite)"
    source = "ClawEval official release (id TBD — verify)"
    tools = ("run_shell", "run_python")
