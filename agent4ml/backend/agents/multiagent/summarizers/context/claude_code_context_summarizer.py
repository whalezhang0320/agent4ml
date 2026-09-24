"""ClaudeCodeContextSummarizer — rule-based extraction (code + review criteria, zero LLM).

设计（design.md §4）:
- per-specialist 输入端转换器
- 从 ThreadState 选取代码 + review 标准（不传全量 context）
- 零 LLM：纯规则提取
"""
from __future__ import annotations

import re
from typing import Any

from agent4ml.backend.agents.state.types import ThreadState

_CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```")
_REVIEW_KEYWORDS = ("review", "check", "verify", "test", "lint", "security", "performance")
_MAX_CODE_BLOCKS = 2
_MAX_SUMMARY_CHARS = 3000


def _field(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


class ClaudeCodeContextSummarizer:
    """Claude Code specialist 输入端转换器（规则提取：代码 + review 标准）。"""

    def summarize(
        self,
        state: ThreadState,
        goal: str,
        success_criteria: str,
        template: Any | None = None,
    ) -> str:
        parts: list[str] = [f"Goal: {goal}", f"Success criteria: {success_criteria}"]

        code_blocks = self._extract_code_blocks(state)
        if code_blocks:
            parts.append("Code to review:")
            parts.extend(code_blocks)

        review_hints = self._extract_review_hints(state)
        if review_hints:
            parts.append(f"Review focus: {', '.join(review_hints)}")

        summary = "\n".join(parts)
        if len(summary) > _MAX_SUMMARY_CHARS:
            summary = summary[:_MAX_SUMMARY_CHARS] + "\n...(truncated)"
        return summary

    def _extract_code_blocks(self, state: ThreadState) -> list[str]:
        blocks: list[str] = []
        for msg in state.get("messages", []):
            content = _field(msg, "content")
            if not isinstance(content, str):
                continue
            found = _CODE_BLOCK_RE.findall(content)
            blocks.extend(found)
            if len(blocks) >= _MAX_CODE_BLOCKS:
                break
        return blocks[:_MAX_CODE_BLOCKS]

    def _extract_review_hints(self, state: ThreadState) -> list[str]:
        hints: list[str] = []
        for msg in state.get("messages", []):
            content = _field(msg, "content")
            if not isinstance(content, str):
                continue
            lower = content.lower()
            for kw in _REVIEW_KEYWORDS:
                if kw in lower and kw not in hints:
                    hints.append(kw)
        return hints[:5]
