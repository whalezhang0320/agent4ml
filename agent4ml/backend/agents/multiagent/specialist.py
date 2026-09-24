"""SpecialistAgent Protocol — 专业 agent 高层抽象。

设计（spec.md SpecialistAgent Requirement + design.md §1）:
- specialist 黑盒：自带 model + 自管 ReAct loop + 自管 context（INV#1, INV#2）
- Agent4ML 只传 goal + context_summary + sandbox_id
- invoke 返 SpecialistRawResult（raw output + artifacts + usage + duration + exit_code）
- 不继承 SpecialistRuntime（SpecialistAgent 是高层，SpecialistRuntime 是低层裸执行）
"""
from __future__ import annotations

from typing import Protocol

from agent4ml.backend.agents.multiagent.types import (
    SpecialistCapabilities,
    SpecialistRawResult,
    SpecialistRequest,
)


class SpecialistAgent(Protocol):
    """专业 agent 契约（黑盒，自带 model + ReAct loop）。

    实现示例：CodexSpecialist / ClaudeCodeSpecialist / SubagentSpecialist（Batch 8）。
    组合 SpecialistRuntime + ContextSummarizer + ResultSummarizer + CredentialProvider。
    """

    @property
    def name(self) -> str:
        """specialist 唯一名（用于 delegate_to_<name> tool 生成 + Registry 注册）。"""
        ...

    @property
    def capabilities(self) -> SpecialistCapabilities:
        """specialist 能力声明（list_specialists 按能力过滤）。"""
        ...

    def invoke(self, request: SpecialistRequest) -> SpecialistRawResult:
        """执行 specialist 调用，返 raw output。

        抛 SpecialistError 子类（Timeout/Crash/Startup/Credential）。
        """
        ...
