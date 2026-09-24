"""ContextSummarizer Protocol — 输入端转换器。

设计（spec.md ContextSummarizer Requirement + design.md §4）:
- per-specialist 实现：每个 specialist 专属 ContextSummarizer
- 从 lead agent 完整 ThreadState 选取相关 context（不传全量，控制 token）
- 生成 context_summary 字符串（传给 specialist，不暴露 specialist 内部）
- 输入端转换器：与 ResultSummarizer（输出端）配对
- L2 扩展：template 参数（L2 启用时传入 is_active 模板，None 时用默认行为）
"""
from __future__ import annotations

from typing import Any, Protocol

from agent4ml.backend.agents.state.types import ThreadState


class ContextSummarizer(Protocol):
    """specialist 输入端转换器契约。

    实现示例：CodexContextSummarizer / ClaudeCodeContextSummarizer /
    SelfCopyContextSummarizer（Batch 7）。
    """

    def summarize(
        self,
        state: ThreadState,
        goal: str,
        success_criteria: str,
        template: Any | None = None,
    ) -> str:
        """从 ThreadState 选取相关 context，返精简 context_summary。

        不传全量 ThreadState（避免污染 specialist + 控制 token 成本）。
        按 specialist 能力定制提取规则（codex 提代码 / claude 提 review 标准 / self-copy 用 LLM）。
        L2 扩展：template 非 None 时按模板 extractors/filters/max_tokens/prompt_skeleton 生成。
        template=None 时用 L1 既有默认行为（L2 未启用，向后兼容）。
        """
        ...
