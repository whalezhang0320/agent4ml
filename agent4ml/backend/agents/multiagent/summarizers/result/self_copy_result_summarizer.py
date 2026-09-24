"""SelfCopyResultSummarizer — base class (no specific extension).

设计（design.md §4）:
- Agent4ML self-copy subagent 输出端转换器
- 使用 base 通用校验（无特定扩展）
- programmatic eval floor（INV#10）
"""
from __future__ import annotations

from agent4ml.backend.agents.multiagent.summarizers.result.base import (
    BaseResultSummarizer,
)


class SelfCopyResultSummarizer(BaseResultSummarizer):
    """Agent4ML self-copy subagent 输出端转换器（基类，无特定扩展）。"""

    def __init__(self) -> None:
        super().__init__(specialist_name="subagent")
