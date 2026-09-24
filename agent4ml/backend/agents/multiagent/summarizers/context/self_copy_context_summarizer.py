"""SelfCopyContextSummarizer — LLM extraction (research context, with rule-based fallback).

设计（design.md §4）:
- per-specialist 输入端转换器
- LLM 提取研究 context（精准提取相关 context）
- 无 LLM 时 fallback 规则提取（recent observations + user_input）
- 不传全量 context（控制 token 成本）
"""
from __future__ import annotations

from typing import Any

from agent4ml.backend.agents.state.types import ThreadState

_MAX_OBSERVATIONS = 5
_MAX_SUMMARY_CHARS = 3000


def _field(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


class SelfCopyContextSummarizer:
    """Agent4ML self-copy subagent 输入端转换器（LLM 提取 + 规则 fallback）。"""

    def __init__(self, llm: Any = None) -> None:
        self._llm = llm

    def summarize(
        self,
        state: ThreadState,
        goal: str,
        success_criteria: str,
        template: Any | None = None,
    ) -> str:
        if self._llm is not None:
            return self._llm_summarize(state, goal, success_criteria)
        return self._rule_summarize(state, goal, success_criteria)

    def _rule_summarize(
        self,
        state: ThreadState,
        goal: str,
        success_criteria: str,
        template: Any | None = None,
    ) -> str:
        parts: list[str] = [f"Goal: {goal}", f"Success criteria: {success_criteria}"]

        user_input = state.get("user_input")
        if user_input:
            parts.append(f"Original request: {user_input}")

        observations = state.get("observations", [])
        recent = observations[-_MAX_OBSERVATIONS:] if observations else []
        for obs in recent:
            content = _field(obs, "content")
            if content:
                parts.append(f"Finding: {content}")

        summary = "\n".join(parts)
        if len(summary) > _MAX_SUMMARY_CHARS:
            summary = summary[:_MAX_SUMMARY_CHARS] + "\n...(truncated)"
        return summary

    def _llm_summarize(
        self,
        state: ThreadState,
        goal: str,
        success_criteria: str,
        template: Any | None = None,
    ) -> str:
        """LLM 提取研究 context（精准提取，MVP 留接口）。"""
        user_input = state.get("user_input", "")
        prompt = (
            f"Extract relevant research context for this subtask.\n"
            f"Goal: {goal}\n"
            f"Success criteria: {success_criteria}\n"
            f"Original request: {user_input}\n"
            f"Return a concise context summary."
        )
        result = self._llm.invoke(prompt)
        content = getattr(result, "content", str(result))
        return str(content)
