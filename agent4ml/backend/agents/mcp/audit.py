"""MCP per-tool-call 审计中间件 — 熔断器 + 审计 + 外化联动标记。

INVARIANT:
- awrap_tool_call 拦截所有工具调用（MCP/builtin/sandbox 统一）
- 熔断器检查：open 不调用，记 circuit_open 事件
- 成功：record_success + 记 ok 事件
- 失败：record_failure + sanitize_error + 记 error 事件
- 外化联动：result 含 AGENT4ML_EXTERNALIZED 标记时加 externalized=true + externalized_path
- 凭证脱敏：error_text 经 sanitizer 清洗后回 LLM
- 无 journal 时静默不记（不报错）
"""
from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command

from agent4ml.backend.agents.middlewares.run_journal_middleware import _get_runtime_value
from agent4ml.backend.agents.middlewares.tagged_context_middleware import (
    AGENT4ML_EXTERNALIZED,
    AGENT4ML_EXTERNALIZED_PATH,
)
from agent4ml.backend.agents.mcp.guards.credential_sanitizer import CredentialSanitizer
from agent4ml.backend.agents.mcp.registry import ToolRegistry

logger = logging.getLogger(__name__)

_BASH_OUTPUT_MAX_CHARS = 10000


def _truncate(text: str, max_chars: int = _BASH_OUTPUT_MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... (truncated, {len(text) - max_chars} chars omitted)"


def _summarize_args(args: Any) -> str:
    """args 摘要（截断防日志爆炸）。"""
    try:
        import json
        s = json.dumps(args, ensure_ascii=False, default=str)
        return _truncate(s, 200)
    except Exception:
        return _truncate(str(args), 200)


class McpAuditMiddleware(AgentMiddleware):
    """per-tool-call 审计 + 熔断器联动。

    INVARIANT:
    - 拦截所有工具调用，统一记 tool.call 事件到 thread-events.jsonl
    - 熔断器 open 时不调用工具，记 circuit_open 事件
    - 凭证脱敏：错误信息经 CredentialSanitizer 清洗后回 LLM
    - 外化联动：result 含 AGENT4ML_EXTERNALIZED 标记时加 externalized 字段
    - 无 journal 时静默不记（不报错）
    """

    def __init__(
        self,
        registry: ToolRegistry,
        sanitizer: CredentialSanitizer | None = None,
    ) -> None:
        self._registry = registry
        self._sanitizer = sanitizer or CredentialSanitizer()

    def _journal(self, runtime: Runtime) -> Any:
        return _get_runtime_value(runtime, "journal", None)

    def _emit(self, runtime: Runtime, event_type: str, payload: dict) -> None:
        journal = self._journal(runtime)
        if journal is not None:
            try:
                journal.append(event_type, payload)
            except Exception as exc:
                logger.debug("journal append failed: %s", exc)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[Any]],
    ) -> Any:
        tool_call = request.tool_call
        tool_name = tool_call.get("name", "")
        runtime = request.runtime

        entry = self._registry.get(tool_name)
        source = entry.source if entry else "unknown"

        start = time.time()

        # 熔断器检查
        if entry and not entry.breaker.allow_call():
            self._audit(
                runtime, tool_name, source, tool_call, None,
                "circuit_open", start,
            )
            return ToolMessage(
                content=f"tool {tool_name} unavailable (circuit open), try alternative",
                tool_call_id=tool_call.get("id", ""),
                status="error",
            )

        try:
            result = await handler(request)
            if entry:
                entry.breaker.record_success()
            self._audit(
                runtime, tool_name, source, tool_call, result,
                "ok", start,
            )
            return result
        except Exception as exc:
            if entry:
                entry.breaker.record_failure()
            sanitized = self._sanitizer.sanitize_error(str(exc))
            self._audit(
                runtime, tool_name, source, tool_call, None,
                "error", start, error=sanitized,
            )
            # 重新抛出，让上层 middleware 处理（ToolCallMiddleware 会转 ToolMessage）
            raise

    def _audit(
        self,
        runtime: Runtime,
        tool_name: str,
        source: str,
        tool_call: dict,
        result: Any,
        status: str,
        start: float,
        error: str | None = None,
    ) -> None:
        """写 tool.call 事件到 thread-events.jsonl。"""
        event: dict[str, Any] = {
            "tool_name": tool_name,
            "source": source,
            "args_summary": _summarize_args(tool_call.get("args")),
            "status": status,
            "duration_ms": int((time.time() - start) * 1000),
        }
        if error:
            event["error"] = error
        # 外化联动标记
        if status == "ok" and isinstance(result, ToolMessage):
            additional_kwargs = getattr(result, "additional_kwargs", {}) or {}
            if additional_kwargs.get(AGENT4ML_EXTERNALIZED):
                event["externalized"] = True
                event["externalized_path"] = additional_kwargs.get(AGENT4ML_EXTERNALIZED_PATH, "")
        self._emit(runtime, "tool.call", event)
