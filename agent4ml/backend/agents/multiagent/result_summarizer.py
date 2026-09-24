"""ResultSummarizer Protocol — 输出端转换器 + programmatic eval。

设计（spec.md ResultSummarizer Requirement + design.md §4）:
- per-specialist 实现：每个 specialist 专属 ResultSummarizer
- 压缩 raw output + 校验 success_criteria（programmatic eval floor，INV#10）
- 生成 gap_analysis（success=False 时从 error + raw_output 提取失败原因）
- 不传回 raw output 全量（控制 lead agent context 膨胀）
- 输出端转换器：与 ContextSummarizer（输入端）配对
- L2 扩展：输出 failure_category（L2 FailureFocuser 读取，D-7=c）
"""
from __future__ import annotations

from typing import Protocol

from agent4ml.backend.agents.multiagent.types import (
    ArtifactRef,
    SpecialistResult,
)


class ResultSummarizer(Protocol):
    """specialist 输出端转换器 + programmatic eval 契约。

    实现示例：BaseResultSummarizer（通用校验）+ CodexResultSummarizer /
    ClaudeCodeResultSummarizer / SelfCopyResultSummarizer（Batch 7）。
    """

    def summarize(
        self,
        raw_output: str,
        artifacts: list[ArtifactRef],
        goal: str,
        success_criteria: str,
    ) -> SpecialistResult:
        """压缩 raw output + 校验 success_criteria，返 SpecialistResult。

        programmatic eval floor（INV#10）：
        - 校验产物存在性 + success_criteria 回应 + 无敏感改动
        - success=False 时 gap_analysis 必填
        - L2 扩展：failure_category 字段（基于 success + raw_output 启发式分类）
        """
        ...
