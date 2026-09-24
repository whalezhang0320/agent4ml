"""SandboxBinder Protocol — 沙箱绑定契约。

设计（proposal.md + design.md §2 shared thread sandbox）:
- shared thread sandbox：lead + self-copy + specialist 同一 sandbox_id（INV#3）
- 模式 B MCP：specialist 通过 SpecialistMcpServer 调 Agent4ML 8 接口（INV#9）
- SandboxBinder 负责 specialist 与 sandbox 的绑定（per-specialist-call）
- 实现示例：PerSubagentBinder（Batch 后续，复用 thread sandbox_id）
"""
from __future__ import annotations

from dataclasses import dataclass

from typing import Protocol


@dataclass(frozen=True)
class BoundSandbox:
    """specialist 沙箱绑定结果（shared thread sandbox，INV#3）。

    sandbox_id 与 lead agent 同一 thread sandbox_id（共享，非物理隔离）。
    specialist_name 标识哪个 specialist 绑定（per-specialist-call 生命周期）。
    """

    sandbox_id: str
    specialist_name: str


class SandboxBinder(Protocol):
    """沙箱绑定契约（specialist 调用前绑定 shared thread sandbox）。

    实现示例：PerSubagentBinder——复用 lead agent thread sandbox_id，
    不创建新 sandbox（shared thread sandbox，INV#3）。
    """

    def bind(self, specialist_name: str, sandbox_id: str) -> BoundSandbox:
        """绑定 specialist 与 sandbox，返 BoundSandbox。

        shared thread sandbox：复用传入的 sandbox_id，不创建新 sandbox。
        """
        ...
