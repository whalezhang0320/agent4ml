"""SubagentSpecialist — 组合 SubagentRuntime，specialist_name=subagent, capability=RESEARCH。

设计（proposal.md + spec.md SpecialistAgent Requirement）:
- Agent4ML self-copy subagent（复用 lead factory，leaf role）
- specialist 黑盒：invoke 只调 runtime（INV#1）
- leaf role 递归控制：tool_groups 不含 multiagent（INV#4）
"""
from __future__ import annotations

from agent4ml.backend.agents.multiagent.runtimes.subagent_runtime import (
    SubagentRuntime,
)
from agent4ml.backend.agents.multiagent.types import (
    SpecialistCapabilities,
    SpecialistCapability,
    SpecialistRawResult,
    SpecialistRequest,
)


class SubagentSpecialist:
    """Agent4ML self-copy subagent specialist（组合 SubagentRuntime，name=subagent, capability=RESEARCH）。"""

    def __init__(self, runtime: SubagentRuntime | None = None) -> None:
        self._runtime = runtime or SubagentRuntime()

    @property
    def name(self) -> str:
        return "subagent"

    @property
    def capabilities(self) -> SpecialistCapabilities:
        return SpecialistCapabilities(
            capabilities=(SpecialistCapability.RESEARCH,),
        )

    def invoke(self, request: SpecialistRequest) -> SpecialistRawResult:
        return self._runtime.invoke(request)
