"""CodexSpecialist — 组合 CodexRuntime，specialist_name=codex, capability=CODING。

设计（proposal.md + spec.md SpecialistAgent Requirement）:
- specialist 黑盒：invoke 只调 runtime，不知 ContextSummarizer/ResultSummarizer（INV#1）
- specialist 自带 model：runtime 启动 codex-acp 自管 ReAct loop（INV#2）
- credential 由 bootstrap 检测（缺失 → specialist 不注册），不进 specialist
"""
from __future__ import annotations

from agent4ml.backend.agents.multiagent.runtimes.codex_runtime import CodexRuntime
from agent4ml.backend.agents.multiagent.types import (
    SpecialistCapabilities,
    SpecialistCapability,
    SpecialistRawResult,
    SpecialistRequest,
)


class CodexSpecialist:
    """Codex specialist（组合 CodexRuntime，name=codex, capability=CODING）。"""

    def __init__(self, runtime: CodexRuntime | None = None) -> None:
        self._runtime = runtime or CodexRuntime()

    @property
    def name(self) -> str:
        return "codex"

    @property
    def capabilities(self) -> SpecialistCapabilities:
        return SpecialistCapabilities(
            capabilities=(SpecialistCapability.CODING,),
        )

    def invoke(self, request: SpecialistRequest) -> SpecialistRawResult:
        return self._runtime.invoke(request)
