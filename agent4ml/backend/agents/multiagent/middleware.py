"""OrchestrationMiddleware — 横切打点 + 产物汇总。

设计（spec.md OrchestrationMiddleware Requirement + design.md §6）:
- awrap_tool_call 拦截 delegate_to_* → 打点四计数器 + 写 ThreadState.orchestration
- 不做编排决策（调谁是 LLM 决定，soft routing）
- specialist 失败转 error ToolMessage（pairing 完整性，INV#6）
- 挂载位置在 ToolCall 后
- set_current_state 供 tool handler 读取 ThreadState
"""
from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any, override

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from agent4ml.backend.agents.multiagent.metrics import MultiAgentMetricsStore
from agent4ml.backend.agents.multiagent.tools import set_current_state
from agent4ml.backend.agents.multiagent.types import ArtifactRef
from agent4ml.backend.agents.state.types import ThreadState


def _tool_text(result: Any) -> str:
    if isinstance(result, ToolMessage):
        content = result.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content if item
            )
        return str(content)
    if isinstance(result, Command):
        messages = result.update.get("messages", [])
        if messages:
            return _tool_text(messages[0])
    return str(result)


class OrchestrationMiddleware(AgentMiddleware):
    """横切打点 + 产物汇总。

    拦截 delegate_to_* tool 调用：
    - Before handler: set_current_state + record_selection + record_invoked
    - After handler: record_completion/record_fallback + write orchestration state
    - specialist 失败转 error ToolMessage（pairing 完整性，INV#6）
    非 delegate_to_* 工具直接 passthrough。
    """

    state_schema = ThreadState  # type: ignore[assignment]

    def __init__(
        self,
        metrics_store: MultiAgentMetricsStore | None = None,
        l2_trigger_middleware: Any | None = None,
        budget_guard: Any | None = None,
    ) -> None:
        self._metrics = metrics_store
        self._l2_trigger_middleware = l2_trigger_middleware
        self._budget_guard = budget_guard

    def _is_delegate_tool(self, tool_name: str) -> bool:
        return tool_name.startswith("delegate_to_")

    def _extract_specialist_name(self, tool_name: str) -> str:
        return tool_name.removeprefix("delegate_to_")

    def _is_success(self, result: Any) -> bool:
        text = _tool_text(result)
        try:
            data = json.loads(text)
            return bool(data.get("success", False))
        except (json.JSONDecodeError, TypeError):
            return False

    def _build_orchestration_update(
        self,
        specialist_name: str,
        result: Any,
    ) -> dict[str, Any]:
        artifacts: list[ArtifactRef] = []
        text = _tool_text(result)
        try:
            data = json.loads(text)
            for a in data.get("artifacts", []):
                artifacts.append(
                    ArtifactRef(
                        path=a.get("path", ""),
                        artifact_type=a.get("type", ""),
                        specialist_name=specialist_name,
                    )
                )
        except (json.JSONDecodeError, TypeError):
            pass
        return {
            "active_specialists": [specialist_name],
            "specialist_artifacts": artifacts,
        }

    def _before_handler(
        self,
        request: ToolCallRequest,
        specialist_name: str,
    ) -> None:
        state = request.state if isinstance(request.state, dict) else {}
        set_current_state(state)
        if self._metrics:
            self._metrics.record_selection(specialist_name)
            self._metrics.record_invoked(specialist_name)

    def _after_handler(
        self,
        specialist_name: str,
        result: Any,
    ) -> dict[str, Any]:
        success = self._is_success(result)
        if self._metrics:
            if success:
                self._metrics.record_completion(specialist_name)
            else:
                self._metrics.record_fallback(specialist_name)
        return self._build_orchestration_update(specialist_name, result)

    def _make_error_response(
        self,
        request: ToolCallRequest,
        exc: Exception,
        specialist_name: str,
    ) -> Command:
        call_id = request.tool_call.get("id", "")
        error_msg = ToolMessage(
            content=json.dumps({
                "success": False,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                "suggestion": "retry, fallback to another specialist, or self-do",
            }),
            tool_call_id=call_id,
            status="error",
        )
        orch = {"active_specialists": [specialist_name], "specialist_artifacts": []}
        return Command(update={"messages": [error_msg], "orchestration": orch})

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        tool_name = request.tool_call.get("name", "")
        if not self._is_delegate_tool(tool_name):
            return handler(request)

        specialist_name = self._extract_specialist_name(tool_name)
        self._before_handler(request, specialist_name)

        try:
            result = handler(request)
        except Exception as exc:
            if self._metrics:
                self._metrics.record_fallback(specialist_name)
            return self._make_error_response(request, exc, specialist_name)

        orch_update = self._after_handler(specialist_name, result)

        if isinstance(result, Command):
            merged = dict(result.update)
            merged["orchestration"] = orch_update
            return Command(update=merged)

        return Command(update={"messages": [result], "orchestration": orch_update})

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[Any]],
    ) -> Any:
        tool_name = request.tool_call.get("name", "")
        if not self._is_delegate_tool(tool_name):
            return await handler(request)

        specialist_name = self._extract_specialist_name(tool_name)
        self._before_handler(request, specialist_name)

        try:
            result = await handler(request)
        except Exception as exc:
            if self._metrics:
                self._metrics.record_fallback(specialist_name)
            return self._make_error_response(request, exc, specialist_name)

        orch_update = self._after_handler(specialist_name, result)

        if isinstance(result, Command):
            merged = dict(result.update)
            merged["orchestration"] = orch_update
            return Command(update=merged)

        return Command(update={"messages": [result], "orchestration": orch_update})
