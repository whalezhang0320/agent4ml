"""HelpRequestMiddleware — intercept ask_help tool calls and pause graph.

When the LLM calls the ask_help tool, this middleware:
1. Intercepts the call before execution
2. Formats a user-friendly help request message
3. Returns Command(goto=END) to pause the graph
4. Writes help.requested to the journal

Borrowed from deer-flow ClarificationMiddleware pattern.
"""

from __future__ import annotations

from typing import Any, override

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.graph import END
from langgraph.types import Command

from agent4ml.backend.agents.middlewares.run_journal_middleware import _get_runtime_value

_HELP_TYPE_ICONS = {
    "missing_info": "❓",
    "approach_choice": "🔀",
    "risk_confirmation": "⚠️",
    "stuck_report": "🚧",
}


def _format_help_message(args: dict[str, Any]) -> str:
    question = args.get("question", "")
    help_type = args.get("help_type", "missing_info")
    context = args.get("context")
    options = args.get("options") or []
    icon = _HELP_TYPE_ICONS.get(help_type, "❓")
    parts = [f"{icon} {context}\n{question}"] if context else [f"{icon} {question}"]
    if options:
        parts.append("")
        for i, opt in enumerate(options, 1):
            parts.append(f"  {i}. {opt}")
    return "\n".join(parts)


class HelpRequestMiddleware(AgentMiddleware):
    """Intercept ask_help tool calls → format → pause graph."""

    @override
    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        tool_call = getattr(request, "tool_call", None) or {}
        if tool_call.get("name") != "ask_help":
            return handler(request)
        return self._handle_help(request, tool_call)

    @override
    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        tool_call = getattr(request, "tool_call", None) or {}
        if tool_call.get("name") != "ask_help":
            return await handler(request)
        return self._handle_help(request, tool_call)

    def _handle_help(self, request: Any, tool_call: dict[str, Any]) -> Command:
        args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id", "")
        message = _format_help_message(args)

        runtime = getattr(request, "runtime", None)
        journal = _get_runtime_value(runtime, "journal", None) if runtime else None
        run_id = _get_runtime_value(runtime, "run_id", None) if runtime else None
        if journal is not None:
            journal.append("help.requested", {
                "run_id": run_id,
                "help_type": args.get("help_type", "missing_info"),
                "question": args.get("question", ""),
            })

        return Command(
            update={"messages": [ToolMessage(
                content=message, tool_call_id=tool_call_id, name="ask_help",
            )]},
            goto=END,
        )
