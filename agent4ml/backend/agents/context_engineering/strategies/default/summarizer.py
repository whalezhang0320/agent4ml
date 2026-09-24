from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from agent4ml.backend.agents.context_engineering.contract import GovernanceResult
from agent4ml.backend.agents.context_engineering.strategies.default._constants import CST
from agent4ml.backend.agents.middlewares.tagged_context_middleware import (
    AGENT4ML_EXTERNALIZED_PATH,
    AGENT4ML_SUMMARY,
)

if TYPE_CHECKING:
    from agent4ml.backend.agents.context_engineering.strategies.default.externalizer import ExternalizerExecutor


class SummarizerExecutor:
    """P4 全量 summarize + pairing 保护。"""

    def __init__(self, model: Any = None, preserve_recent: int = 6) -> None:
        self._model = model
        self._preserve_recent = preserve_recent

    def summarize_if_pending(self, governance: dict | None, messages: list, externalizer: ExternalizerExecutor) -> GovernanceResult | None:
        governance = governance or {}
        pending = (governance.get("default") or {}).get("pending") or []
        if "P4" not in pending:
            return None
        if not self._model:
            return None
        to_summarize, preserved = self._partition(messages)
        if not to_summarize:
            return None
        externalized_paths = self._externalize_orphans(to_summarize, externalizer)
        summary_text = self._call_llm(to_summarize)
        if summary_text is None:
            summary_text = "压缩失败，保留最近对话。"
        if externalized_paths:
            summary_text += "\n\n外化工具结果：" + ", ".join(externalized_paths)
        summary_msg = HumanMessage(content=summary_text, additional_kwargs={AGENT4ML_SUMMARY: True})
        return GovernanceResult(
            state_patch={"governance": self._update_summary(governance, summary_text)},
            messages_patch=[RemoveMessage(id=REMOVE_ALL_MESSAGES), summary_msg, *preserved],
        )

    def _partition(self, messages: list) -> tuple[list, list]:
        n = len(messages)
        if n <= self._preserve_recent:
            return [], messages
        cut = n - self._preserve_recent
        cut = self._snap_to_pairing(messages, cut)
        to_summarize = list(messages[:cut])
        preserved = list(messages[cut:])
        # preserved 孤立 ToolMessage（配对 AIMessage 在 to_summarize）移到 to_summarize，防 400
        preserved, orphans = self._strip_orphan_tools(preserved)
        to_summarize.extend(orphans)
        return to_summarize, preserved

    @staticmethod
    def _strip_orphan_tools(preserved: list) -> tuple[list, list]:
        """preserved 中孤立 ToolMessage 或孤立 AIMessage(tool_calls) 移除，防 pairing 断裂。"""
        ai_tc_ids: set[str] = set()
        for msg in preserved:
            if isinstance(msg, AIMessage):
                for tc in msg.tool_calls or []:
                    tc_id = tc.get("id") if isinstance(tc, dict) else None
                    if tc_id:
                        ai_tc_ids.add(tc_id)
        tool_ids: set[str] = set()
        for msg in preserved:
            if isinstance(msg, ToolMessage):
                tool_ids.add(msg.tool_call_id)
        clean: list = []
        orphans: list = []
        for msg in preserved:
            is_orphan_tool = isinstance(msg, ToolMessage) and msg.tool_call_id not in ai_tc_ids
            is_orphan_ai = isinstance(msg, AIMessage) and bool(msg.tool_calls) and not all(
                (tc.get("id") if isinstance(tc, dict) else None) in tool_ids
                for tc in msg.tool_calls
            )
            if is_orphan_tool or is_orphan_ai:
                orphans.append(msg)
            else:
                clean.append(msg)
        return clean, orphans

    def _snap_to_pairing(self, messages: list, cut: int) -> int:
        while cut < len(messages) and isinstance(messages[cut], ToolMessage):
            if cut > 0 and isinstance(messages[cut - 1], AIMessage) and messages[cut - 1].tool_calls:
                cut -= 1
            else:
                break
        return cut

    def _externalize_orphans(self, messages: list, externalizer: ExternalizerExecutor) -> list[str]:
        ai_tc_ids: set[str] = set()
        for msg in messages:
            if isinstance(msg, AIMessage):
                for tc in msg.tool_calls or []:
                    tc_id = tc.get("id") if isinstance(tc, dict) else None
                    if tc_id:
                        ai_tc_ids.add(tc_id)
        paths: list[str] = []
        for msg in messages:
            if isinstance(msg, ToolMessage) and msg.tool_call_id not in ai_tc_ids:
                rewritten = externalizer.externalize_if_needed(msg)
                if rewritten is not None:
                    path = rewritten.additional_kwargs.get(AGENT4ML_EXTERNALIZED_PATH)
                    if path:
                        paths.append(path)
        return paths

    def _call_llm(self, messages: list) -> str | None:
        if not self._model:
            return None
        try:
            from agent4ml.backend.agents.prompts import get_prompt_manager

            history = self._format_history(messages)
            prompt = get_prompt_manager().load("context_engineering/default", "summarize", messages_text=history)
            # config tags 标记内部调用，防 astream(stream_mode=messages) 捕获泄漏到 CLI
            response = self._model.invoke(prompt, config={"tags": ["internal_llm"]})
            return response.text.strip() if hasattr(response, "text") else str(response)
        except Exception:
            return None

    @staticmethod
    def _format_history(messages: list) -> str:
        lines: list[str] = []
        for msg in messages:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            lines.append(f"[{type(msg).__name__}] {content[:500]}")
        return "\n".join(lines)

    def _update_summary(self, governance: dict, summary_text: str) -> dict:
        g = dict(governance or {})
        d = dict(g.get("default") or {})
        d["summary"] = summary_text
        d["summary_id"] = "summary_" + datetime.now(CST).strftime("%H%M%S%f")
        metrics = dict(d.get("metrics") or {})
        metrics["summarize_count"] = metrics.get("summarize_count", 0) + 1
        d["metrics"] = metrics
        g["default"] = d
        return g
