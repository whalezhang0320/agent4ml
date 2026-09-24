"""SubagentProvider Protocol — Agent4ML self-copy subagent 契约。

设计（spec.md SubagentProvider Requirement + design.md §2）:
- Agent4ML self-copy：复用 lead agent factory（create_agent4ml_agent）
- leaf role 递归控制：子 agent tool_groups 不含 multiagent，不能 spawn（INV#4）
- isolated context：全新 ThreadState，不继承父 message history
- shared thread sandbox：复用父 sandbox_id（INV#3）
- sync only MVP：spawn 同步返回（INV#5）

与 SpecialistRuntime 区别：
- SubagentProvider.spawn(SubagentRequest) → SubagentResult（高层 API）
- SubagentRuntime.invoke(SpecialistRequest) → SpecialistRawResult（低层 API，实现 SpecialistRuntime）
- SubagentSpecialist 组合 SubagentRuntime（适配 SubagentProvider 为 SpecialistRuntime）
"""
from __future__ import annotations

from typing import Protocol

from agent4ml.backend.agents.multiagent.types import (
    SubagentRequest,
    SubagentResult,
)


class SubagentProvider(Protocol):
    """Agent4ML self-copy subagent 契约（sync only，leaf role）。

    实现示例：Batch 6 SubagentRuntime 内构造 lead factory 调用。
    """

    def spawn(self, request: SubagentRequest) -> SubagentResult:
        """spawn 一个 Agent4ML self-copy subagent 执行任务。

        leaf role（INV#4）：子 agent tool_groups 不含 multiagent，看不到 delegate_to_* tool。
        isolated context：全新 ThreadState，只传 goal + context_summary。
        shared thread sandbox：复用父 sandbox_id（INV#3）。

        抛 SubagentError 子类（SubagentTimeoutError / SubagentMaxStepsError）。
        """
        ...
