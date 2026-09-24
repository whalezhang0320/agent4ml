"""ClaudeCodeSpecialist — 组合 ClaudeCodeRuntime，specialist_name=claude, capability=REVIEW。

设计（proposal.md + spec.md SpecialistAgent Requirement）:
- specialist 黑盒：invoke 只调 runtime（INV#1）
- specialist 自带 model：claude CLI 自管 ReAct loop（INV#2）
"""
from __future__ import annotations

from agent4ml.backend.agents.multiagent.runtimes.claude_code_runtime import (
    ClaudeCodeRuntime,
)
from agent4ml.backend.agents.multiagent.types import (
    SpecialistCapabilities,
    SpecialistCapability,
    SpecialistRawResult,
    SpecialistRequest,
)


class ClaudeCodeSpecialist:
    """Claude Code specialist（组合 ClaudeCodeRuntime，name=claude, capability=REVIEW）。"""

    def __init__(self, runtime: ClaudeCodeRuntime | None = None) -> None:
        self._runtime = runtime or ClaudeCodeRuntime()

    @property
    def name(self) -> str:
        return "claude"

    @property
    def capabilities(self) -> SpecialistCapabilities:
        return SpecialistCapabilities(
            capabilities=(SpecialistCapability.REVIEW,),
        )

    def invoke(self, request: SpecialistRequest) -> SpecialistRawResult:
        return self._runtime.invoke(request)
