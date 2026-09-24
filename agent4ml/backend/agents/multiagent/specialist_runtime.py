"""SpecialistRuntime Protocol — 裸执行契约（sync only MVP）。

设计（spec.md SpecialistRuntime Requirement + design.md §2）:
- sync only MVP：只定义 invoke，不做 async ainvoke（INV#5）
- 只负责跑 specialist loop 返 raw output
- 不知 ContextSummarizer / ResultSummarizer / ThreadState（低层，纯执行）
- 异常统一：抛 SpecialistError 子类
"""
from __future__ import annotations

from typing import Protocol

from agent4ml.backend.agents.multiagent.types import (
    SpecialistRawResult,
    SpecialistRequest,
)


class SpecialistRuntime(Protocol):
    """specialist runtime 裸执行契约（sync only）。

    实现示例：CodexRuntime / ClaudeCodeRuntime / SubagentRuntime（Batch 6）。
    """

    def invoke(self, request: SpecialistRequest) -> SpecialistRawResult:
        """执行 specialist loop，返 raw output。

        抛 SpecialistError 子类（SpecialistTimeoutError / SpecialistCrashError /
        SpecialistStartupError / SpecialistCredentialError）。
        """
        ...
