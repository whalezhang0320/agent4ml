"""DanglingToolCallMiddleware — patch interrupted tool calls on resume.

When the graph resumes after a help request or user interrupt, the last
AIMessage may contain tool_calls without corresponding ToolMessages
(dangling calls). This causes LLM 400 errors due to incomplete message
format. This middleware scans message history in before_model and injects
synthetic error ToolMessages for each dangling call.

Borrowed from deer-flow DanglingToolCallMiddleware pattern.
"""

from __future__ import annotations

import json
from typing import Any, override

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.runtime import Runtime

from agent4ml.backend.agents.state.types import ThreadState

_MAX_RECOVERY_DETAIL_LEN = 500


class DanglingToolCallMiddleware(AgentMiddleware):
    """Inject placeholder ToolMessages for dangling tool calls before model invocation."""

    state_schema = ThreadState  # type: ignore[assignment]

    @staticmethod
    def _extract_tool_calls(msg: AIMessage) -> list[dict[str, Any]]:
        calls = list(getattr(msg, "tool_calls", None) or [])
        raw = (getattr(msg, "additional_kwargs", None) or {}).get("tool_calls") or []
        if not calls:
            for rtc in raw:
                if not isinstance(rtc, dict):
                    continue
                fn = rtc.get("function", {})
                name = rtc.get("name") or fn.get("name", "unknown")
                args = rtc.get("args", {})
                if not args and isinstance(fn, dict):
                    raw_args = fn.get("arguments", "{}")
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                calls.append({"id": rtc.get("id", ""), "name": name, "args": args})
        return calls

    @override
    def before_model(self, state: ThreadState, runtime: Runtime) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        if not messages:
            return None

        patch: list[ToolMessage] = []
        answered_ids: set[str] = set()
        for msg in messages:
            if isinstance(msg, ToolMessage):
                answered_ids.add(msg.tool_call_id)

        for msg in messages:
            if not isinstance(msg, AIMessage):
                continue
            calls = self._extract_tool_calls(msg)
            for tc in calls:
                tc_id = tc.get("id", "")
                if tc_id and tc_id not in answered_ids:
                    patch.append(ToolMessage(
                        content="[Tool call was interrupted and did not return a result.]",
                        tool_call_id=tc_id,
                        name=tc.get("name", "unknown"),
                    ))
                    answered_ids.add(tc_id)

        if not patch:
            return None
        return {"messages": patch}

    @override
    async def abefore_model(self, state: ThreadState, runtime: Runtime) -> dict[str, Any] | None:
        return self.before_model(state, runtime)
