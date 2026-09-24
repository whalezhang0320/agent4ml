"""标签化上下文格式基座 — TaggedContextMiddleware + ContextAssembler。

送 LLM 的上下文渲染为扁平 XML 标签序列 + <turn> 时序分组。request-scoped
不持久，state 存原始 message content + additional_kwargs 标记。

层 2 标记 agent4ml.* 命名空间服务于压缩筛选 + 幂等 + trace 审计。

分 batch 实现：
- Batch 1：标记常量（本文件）
- Batch 2：ContextAssembler 头部上下文块渲染
- Batch 3：ContextAssembler message 渲染 + turn 分组
- Batch 4：TaggedContextMiddleware wrap_model_call
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping, override

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.runtime import Runtime

from agent4ml.backend.agents.state.types import ThreadState

logger = logging.getLogger(__name__)

# 层 2 标记键名（additional_kwargs 命名空间 agent4ml.*）
# 服务于压缩筛选 + 幂等 + trace 审计。命名空间隔离防与其他 middleware 冲突。
AGENT4ML_THINKING = "agent4ml.thinking"
AGENT4ML_SUMMARY = "agent4ml.summary"
AGENT4ML_EXTERNALIZED = "agent4ml.externalized"
AGENT4ML_EXTERNALIZED_PATH = "agent4ml.externalized_path"
AGENT4ML_EXTERNALIZED_META = "agent4ml.externalized_meta"
AGENT4ML_COMPACTION_STAGE = "agent4ml.compaction_stage"
AGENT4ML_TURN_ID = "agent4ml.turn_id"

_CST = timezone(timedelta(hours=8))


def _field(item: Any, name: str) -> Any:
    """从 dict / dataclass / obj 取字段，统一访问。"""
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


class ContextAssembler:
    """渲染扁平 XML 标签序列。request-scoped，不持久。

    render_context_block：头部上下文块（state 字段 → <goal><plan><reflection>
    <summary><date>）。<system> 在 wrap_model_call 从 request SystemMessage 提取。
    <observations> 舍弃（Q5，留 state 供 ReflectionMiddleware）。
    """

    def __init__(self, max_reflections: int = 10) -> None:
        self._max_reflections = max_reflections

    def render_context_block(self, state: Mapping[str, Any], governance: dict | None) -> str:
        """渲染头部上下文块。从 ThreadState 字段 + governance 渲染。"""
        lines: list[str] = []

        goal = state.get("research_question") or ""
        if goal:
            lines.append(f"<goal>{goal}</goal>")

        plan = self._render_plan(state.get("todos"))
        if plan:
            lines.append(f"<plan>\n{plan}\n</plan>")

        reflection = self._render_reflection(state)
        if reflection:
            lines.append(f"<reflection>\n{reflection}\n</reflection>")

        summary = self._render_summary(governance)
        if summary:
            lines.append(f"<summary>\n{summary}\n</summary>")

        lines.append(f"<date>{self._current_date()}</date>")
        return "\n".join(lines)

    def _render_plan(self, todos: list | None) -> str:
        if not todos:
            return ""
        items: list[str] = []
        for t in todos:
            title = _field(t, "title") or _field(t, "description") or str(t)
            status = _field(t, "status") or ""
            mark = {"completed": "x", "in_progress": ">"}.get(status, " ")
            items.append(f"[{mark}] {title}")
        return "\n".join(items)

    def _render_summary(self, governance: dict | None) -> str:
        if not governance:
            return ""
        default = governance.get("default") or {}
        return default.get("summary") or ""

    def _render_reflection(self, state: Mapping[str, Any]) -> str:
        items = state.get("reflection_items") or []
        if not items:
            return ""
        recent = list(items)[-self._max_reflections:]
        lines: list[str] = []
        for item in recent:
            scope = _field(item, "scope") or ""
            kind = _field(item, "kind") or ""
            question = _field(item, "question") or str(item)
            status = _field(item, "status") or "open"
            lines.append(f"- [{status}] {scope}/{kind}: {question}")
        return "\n".join(lines)

    def _current_date(self) -> str:
        return datetime.now(_CST).strftime("%Y-%m-%d, %A")

    def render_messages(self, messages: list) -> str:
        """渲染 message 序列为 <turn> 时序分组标签。

        - HumanMessage（普通）开新 <turn> + <message role="user">
        - HumanMessage agent4ml.summary 跳过（context_block 已渲染）
        - SystemMessage 跳过（wrap_model_call 从 request 提取，Batch 4）
        - AIMessage：agent4ml.thinking → <thinking>；tool_calls → <toolcall>；content → <answer>
        - ToolMessage：agent4ml.externalized → <toolresult path=...>；普通 → <toolresult>
        """
        lines: list[str] = []
        turn_id = 0
        turn_open = False

        for msg in messages:
            if isinstance(msg, SystemMessage):
                continue
            if isinstance(msg, HumanMessage) and msg.additional_kwargs.get(AGENT4ML_SUMMARY):
                continue

            if isinstance(msg, HumanMessage):
                if turn_open:
                    lines.append("</turn>")
                turn_id += 1
                lines.append(f'<turn id="{turn_id}">')
                turn_open = True
                lines.append(f'<message role="user">{self._escape(self._extract_text(msg))}</message>')
                continue

            if isinstance(msg, AIMessage):
                if msg.additional_kwargs.get(AGENT4ML_THINKING):
                    thinking = msg.additional_kwargs.get("reasoning_content") or ""
                    if thinking:
                        lines.append(f"<thinking>{self._escape(thinking)}</thinking>")
                for tc in msg.tool_calls or []:
                    name = tc.get("name", "") if isinstance(tc, dict) else ""
                    args = str(tc.get("args", "") if isinstance(tc, dict) else "")[:100]
                    lines.append(f'<toolcall name="{self._escape(name)}" args="{self._escape(args)}"/>')
                content = self._extract_text(msg)
                if content:
                    lines.append(f"<answer>{self._escape(content)}</answer>")
                continue

            if isinstance(msg, ToolMessage):
                name = msg.name or "unknown"
                content = self._extract_text(msg)
                if msg.additional_kwargs.get(AGENT4ML_EXTERNALIZED):
                    path = msg.additional_kwargs.get(AGENT4ML_EXTERNALIZED_PATH, "")
                    meta = msg.additional_kwargs.get(AGENT4ML_EXTERNALIZED_META) or {}
                    tokens = meta.get("tokens_saved", "")
                    lines.append(
                        f'<toolresult name="{self._escape(name)}" path="{self._escape(path)}" tokens="{tokens}">'
                        f"{self._escape(content)}</toolresult>"
                    )
                else:
                    lines.append(f'<toolresult name="{self._escape(name)}">{self._escape(content)}</toolresult>')
                continue

        if turn_open:
            lines.append("</turn>")
        return "\n".join(lines)

    @staticmethod
    def _extract_text(message: Any) -> str:
        """从 message（str / BaseMessage / content）提取纯文本。"""
        if isinstance(message, str):
            return message
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    parts.append(part.get("text", ""))
            return "".join(parts)
        return str(content) if content is not None else ""

    @staticmethod
    def _escape(text: str) -> str:
        """转义 XML 特殊字符，防标签结构破坏。"""
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def render_messages_for_llm(self, messages: list) -> list:
        """返改写后 messages 列表（方案 H：角色保留 + AIMessage 语义标签 + ToolMessage 经典）。

        - SystemMessage：跳过（提取进上下文块 <system>）
        - HumanMessage agent4ml.summary：跳过（<summary> 已在上下文块）
        - HumanMessage 普通：原生不变
        - AIMessage：content 包 <thinking>...</thinking><answer>...</answer>
        - ToolMessage：经典不变
        """
        rewritten: list = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                continue
            if isinstance(msg, HumanMessage):
                if msg.additional_kwargs.get(AGENT4ML_SUMMARY):
                    continue
                rewritten.append(msg)
                continue
            if isinstance(msg, AIMessage):
                rewritten.append(self._rewrite_ai_message(msg))
                continue
            rewritten.append(msg)
        return rewritten

    def _rewrite_ai_message(self, msg: AIMessage) -> AIMessage:
        """AIMessage content 包 <thinking><answer> 语义标签（机制层 tool_calls 不变）。"""
        parts: list[str] = []
        if msg.additional_kwargs.get(AGENT4ML_THINKING):
            thinking = msg.additional_kwargs.get("reasoning_content") or ""
            if thinking:
                parts.append(f"<thinking>{self._escape(thinking)}</thinking>")
        content = self._extract_text(msg)
        if content:
            parts.append(f"<answer>{self._escape(content)}</answer>")
        new_content = "".join(parts) if parts else msg.content
        return msg.model_copy(update={"content": new_content})


class TaggedContextMiddleware(AgentMiddleware):
    """标签化上下文基座 middleware。

    wrap_model_call 调 ContextAssembler 渲染扁平标签序列，request.override
    送 LLM。request-scoped 不持久（state 存原始 message content）。
    trace 经 logger 记录；state.tagged_context 字段留接口，后续补持久快照。
    """

    state_schema = ThreadState  # type: ignore[assignment]

    def __init__(self, assembler: ContextAssembler | None = None) -> None:
        self._assembler = assembler or ContextAssembler()

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        assembled = self._assemble(request)
        logger.info("tagged_context assembled: %d messages", len(assembled.messages))
        return handler(assembled)

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        assembled = self._assemble(request)
        logger.info("tagged_context assembled: %d messages", len(assembled.messages))
        return await handler(assembled)

    def _assemble(self, request: ModelRequest) -> ModelRequest:
        """方案 H：上下文块 SystemMessage + 改写后对话历史（角色保留）。"""
        state = getattr(getattr(request, "runtime", None), "state", None) or {}
        messages = getattr(request, "messages", None) or []
        system_text = self._extract_system(messages)
        governance = state.get("governance")
        context_block = self._assembler.render_context_block(state, governance)
        if system_text:
            context_block = f"<system>\n{system_text}\n</system>\n\n{context_block}"
        rewritten = self._assembler.render_messages_for_llm(messages)
        new_messages: list = []
        if context_block:
            new_messages.append(SystemMessage(content=context_block))
        new_messages.extend(rewritten)
        return request.override(messages=new_messages)

    @override
    def after_model(self, state: ThreadState, runtime: Runtime) -> dict[str, Any] | None:
        """trace 审计：从 state.messages 重组渲染写 state.tagged_context。"""
        messages = state.get("messages") or []
        governance = state.get("governance")
        system_text = self._extract_system(messages)
        context_block = self._assembler.render_context_block(state, governance)
        if system_text:
            context_block = f"<system>\n{system_text}\n</system>\n\n{context_block}"
        messages_trace = self._assembler.render_messages(messages)
        trace = (context_block + "\n\n" + messages_trace) if context_block else messages_trace
        return {"tagged_context": {"rendered": trace, "created_at": datetime.now(_CST).isoformat()}}

    @staticmethod
    def _extract_system(messages: list) -> str:
        """提取原 SystemMessage content 作为 <system> 标签内容。"""
        for msg in messages:
            if isinstance(msg, SystemMessage):
                content = msg.content
                return content if isinstance(content, str) else str(content)
        return ""
