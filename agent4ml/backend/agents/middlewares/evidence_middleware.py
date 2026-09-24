"""EvidenceMiddleware — 拦截证据类工具调用，沉淀 Source/Observation/AgentError 进 ThreadState。

激活 observations/sources/errors 死字段。仅 wrap_tool_call hook，不改 ReAct 内核。
工具结果双写：messages（模型可见）+ observations/sources（旁路存档）。
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any, override

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command

from agent4ml.backend.agents.state.types import Observation, Source, ThreadState

# 证据类工具白名单（D9：MVP 手维护）。非白名单工具直接 passthrough。
_EVIDENCE_TOOLS: frozenset[str] = frozenset({
    "web_search",
    "deep_search",
    "browse_page",
    "get_page_links",
    "github_search",
    "github_repo_files",
})

_OBS_CONTENT_MAX = 800
_URL_RE = re.compile(r"https?://[^\s\)\]\}\>\"']+")


def _make_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _tool_text(tool_msg: ToolMessage) -> str:
    """Normalize ToolMessage content (str | list[dict]) to plain text."""
    content = tool_msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content)


def _extract_sources(tool_name: str, tool_msg: ToolMessage) -> list[Source]:
    """从 ToolMessage 文本启发式抽取 URL → Source（去重按 url）。"""
    text = _tool_text(tool_msg)
    seen: set[str] = set()
    sources: list[Source] = []
    for match in _URL_RE.finditer(text):
        url = match.group(0).rstrip(".,;:")
        if url in seen:
            continue
        seen.add(url)
        sources.append(Source(
            source_id=_make_id("src"),
            url=url,
            title="",
            source_type="web",
            retrieved_at=_now_iso(),
            summary="",
        ))
    return sources


def _make_observation(
    tool_name: str,
    tool_msg: ToolMessage,
    sources: list[Source],
    step_id: str | None,
) -> Observation:
    """裁剪正文 ~800 字 → Observation，source_refs 关联本次 Source。"""
    text = _tool_text(tool_msg)
    content = text[:_OBS_CONTENT_MAX]
    return Observation(
        observation_id=_make_id("obs"),
        step_id=step_id,
        content=content,
        source_refs=tuple(s.source_id for s in sources),
        created_at=_now_iso(),
    )


def _resolve_step_id(state: Any) -> str | None:
    """F4：取 current_step_id；无则从 todos 派生兜底（首个 in_progress 或 todo-0），避免 None。

    首轮模型可能直接调搜索（未先 write_todos）→ current_step_id 缺失 → step_id=None
    会导致 Reflection 覆盖度误判。兜底关联到最近 in_progress todo 或 todo-0。
    """
    if not isinstance(state, dict):
        return None
    step_id = state.get("current_step_id")
    if step_id:
        return step_id
    todos = state.get("todos") or []
    if not todos:
        return None
    # 优先首个 in_progress todo 的 index
    for idx, t in enumerate(todos):
        if isinstance(t, dict) and t.get("status") == "in_progress":
            return f"todo-{idx}"
    # 无 in_progress 则兜底 todo-0
    return "todo-0"


class EvidenceMiddleware(AgentMiddleware):
    """拦截证据类工具调用，把结果结构化沉淀进 observations/sources/errors。

    非证据工具直接 passthrough。证据工具返回 Command(update=...) 双写。
    """

    state_schema = ThreadState  # type: ignore[assignment]

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        tool_name = request.tool_call.get("name", "")
        if tool_name not in _EVIDENCE_TOOLS:
            return handler(request)

        # FD17：Evidence 只管证据抽取（成功时）；失败分类/账本归 ToolCallMiddleware（外层捕获异常）
        result = handler(request)
        if not isinstance(result, ToolMessage):
            return result

        from agent4ml.backend.agents.observability.interrupt_protection import (
            interrupt_protection,
        )
        with interrupt_protection():
            sources = _extract_sources(tool_name, result)
            state = request.state
            step_id = _resolve_step_id(state)
            obs = _make_observation(tool_name, result, sources, step_id)
        return Command(update={
            "observations": [obs],
            "sources": sources,
            "messages": [result],
        })

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        tool_name = request.tool_call.get("name", "")
        if tool_name not in _EVIDENCE_TOOLS:
            return await handler(request)

        result = await handler(request)
        if not isinstance(result, ToolMessage):
            return result

        from agent4ml.backend.agents.observability.interrupt_protection import (
            interrupt_protection,
        )
        with interrupt_protection():
            sources = _extract_sources(tool_name, result)
            state = request.state
            step_id = _resolve_step_id(state)
            obs = _make_observation(tool_name, result, sources, step_id)
        return Command(update={
            "observations": [obs],
            "sources": sources,
            "messages": [result],
        })
