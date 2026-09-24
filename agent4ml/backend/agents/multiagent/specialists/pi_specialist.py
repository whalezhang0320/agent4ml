"""PiSpecialist — 组合 PiRuntime，specialist_name=pi, capability=CODING。

设计（proposal.md + spec.md PiSpecialist Requirement + design_docs/46 §10.3.5）:
- specialist 黑盒：invoke 只调 runtime，不知 ContextSummarizer/ResultSummarizer（INV#1）
- specialist 自带 model：pi CLI 自管 ReAct loop（INV#2）
- credential 由 bootstrap 检测（缺失 → specialist 不注册），不进 specialist
- 决策 1：强制走 Agent4ML SpecialistMcpServer（pi --no-builtin-tools + -e agent4ml-sandbox-bridge）
"""
from __future__ import annotations

from typing import Any

from agent4ml.backend.agents.multiagent.runtimes.pi_runtime import (
    PiRuntime,
    PiRuntimeConfig,
)
from agent4ml.backend.agents.multiagent.types import (
    SpecialistCapabilities,
    SpecialistCapability,
    SpecialistRawResult,
    SpecialistRequest,
)


class PiSpecialist:
    """Pi coding specialist（组合 PiRuntime，name=pi, capability=CODING）。

    与 CodexSpecialist / ClaudeCodeSpecialist 并列，作为第四个 specialist。
    决策 4（设计文档 46 §10.6）：按任务类型路由——coding → pi preferred。
    """

    def __init__(
        self,
        runtime: PiRuntime | None = None,
        credential: Any | None = None,
    ) -> None:
        self._runtime = runtime or PiRuntime()
        self._credential = credential  # 凭证由 bootstrap 检测，传给 specialist 仅供 runtime 用

    @property
    def name(self) -> str:
        return "pi"

    @property
    def capabilities(self) -> SpecialistCapabilities:
        return SpecialistCapabilities(
            capabilities=(SpecialistCapability.CODING,),
        )

    def invoke(self, request: SpecialistRequest) -> SpecialistRawResult:
        return self._runtime.invoke(request)
