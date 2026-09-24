"""RunJournalMiddleware — 把 agent/model/tool 生命周期事件记入 RunJournal。

AgentMiddleware 版（取代旧 BaseMiddleware 体系）。从 runtime configurable 读
journal / run_id（经 _get_runtime_value 三路径，兼容 LangGraph ≥1.1.9 与测试 SimpleNamespace）。
_get_runtime_value 为模块级函数，供 TodoMiddleware / _jump_budget 复用。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, override

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime


def _get_runtime_value(runtime: Any, key: str, default: Any = None) -> Any:
    """提取 runtime configurable 值。

    路径（按优先级）：
    1. runtime.context（dict/对象）—— 有 context_schema 时
    2. langgraph.config.get_config()["configurable"]—— create_agent 标准 hook 内访问 config 的方式
    3. runtime.get_configurable()—— LangGraph runtime API（如有）
    4. runtime.config["configurable"]—— 兜底
    """
    if runtime is not None:
        # Path 1: runtime.context
        ctx = getattr(runtime, "context", None)
        if ctx is not None:
            if isinstance(ctx, dict):
                if key in ctx:
                    return ctx[key]
            else:
                val = getattr(ctx, key, None)
                if val is not None:
                    return val
    # Path 2: get_config() —— create_agent hook 内取 configurable 的标准方式（F2 修复）
    try:
        from langgraph.config import get_config

        config = get_config()
        if config:
            configurable = config.get("configurable", {})
            if isinstance(configurable, dict) and key in configurable:
                return configurable[key]
    except Exception:
        pass
    # Path 3: runtime.get_configurable()
    if runtime is not None:
        get_cfg = getattr(runtime, "get_configurable", None)
        if callable(get_cfg):
            try:
                cfg = get_cfg()
                if cfg and isinstance(cfg, dict) and key in cfg:
                    return cfg[key]
            except Exception:
                pass
        # Path 4: runtime.config["configurable"]
        config = getattr(runtime, "config", None)
        if isinstance(config, dict):
            configurable = config.get("configurable", {})
            if isinstance(configurable, dict) and key in configurable:
                return configurable[key]
    return default


def _tool_text(result: Any) -> str:
    if isinstance(result, ToolMessage):
        content = result.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                item["text"] if isinstance(item, dict) and "text" in item else str(item)
                for item in content if item
            )
        return str(content)
    return str(result)


def _result_status(result: Any) -> str:
    """Inspect tool result for error indicators beyond Python exceptions.

    Tools may return ``ToolMessage(status="error")`` or ``Command(update={
    'errors': [{'kind': 'failure'}], 'messages': [ToolMessage(status='error')]})``
    without raising. The journal must record these as ``status="error"``.
    """
    if isinstance(result, ToolMessage):
        if getattr(result, "status", None) == "error":
            return "error"
        return "ok"
    update = getattr(result, "update", None)
    if isinstance(update, dict):
        errors = update.get("errors")
        if isinstance(errors, list):
            for err in errors:
                if isinstance(err, dict) and err.get("kind") == "failure":
                    return "error"
        messages = update.get("messages")
        if isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, ToolMessage) and getattr(msg, "status", None) == "error":
                    return "error"
    return "ok"


def _truncate_tool_input(tool_input: Any) -> str:
    """Short summary of tool input for activity display."""
    if isinstance(tool_input, dict):
        cmd = tool_input.get("command") or tool_input.get("path") or str(tool_input)
        return str(cmd)[:80]
    return str(tool_input)[:80]


class RunJournalMiddleware(AgentMiddleware):
    """记录 agent/model/tool 事件到 RunJournal。无 journal 时静默不报错。"""

    def _journal(self, runtime: Runtime) -> Any:
        return _get_runtime_value(runtime, "journal", None)

    def _run_id(self, runtime: Runtime) -> Any:
        return _get_runtime_value(runtime, "run_id", None)

    def _activity_tracker(self, runtime: Runtime) -> Any:
        return _get_runtime_value(runtime, "activity_tracker", None)

    def _model_name(self, runtime: Runtime) -> str:
        return str(_get_runtime_value(runtime, "model", "") or "")

    @override
    def before_agent(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        journal = self._journal(runtime)
        if journal is not None:
            journal.append("agent.started", {"run_id": self._run_id(runtime)})
        return None

    @override
    def after_agent(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        journal = self._journal(runtime)
        if journal is not None:
            journal.append("agent.finished", {"run_id": self._run_id(runtime)})
        return None

    @override
    def before_model(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        journal = self._journal(runtime)
        if journal is not None:
            journal.append("llm.request", {"run_id": self._run_id(runtime), "model": self._model_name(runtime)})
        return None

    @override
    def after_model(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        journal = self._journal(runtime)
        if journal is not None:
            journal.append("llm.response", {"run_id": self._run_id(runtime), "model": self._model_name(runtime)})
        return None

    @override
    def wrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Any],
    ) -> Any:
        runtime = getattr(request, "runtime", None)
        journal = self._journal(runtime) if runtime is not None else None
        run_id = self._run_id(runtime) if runtime is not None else None
        tool_call = getattr(request, "tool_call", None) or {}
        tool_name = tool_call.get("name", "") if isinstance(tool_call, dict) else ""
        tool_input = tool_call.get("args", {}) if isinstance(tool_call, dict) else {}

        if journal is not None:
            journal.append("tool.called", {
                "run_id": run_id,
                "tool_name": tool_name,
                "tool_input": tool_input,
            })
        tracker = self._activity_tracker(runtime) if runtime is not None else None
        activity_id = f"{tool_name}:{tool_call.get('id', '')}" if isinstance(tool_call, dict) else tool_name
        if tracker is not None:
            tracker.start(activity_id, "tool", f"{tool_name}: {_truncate_tool_input(tool_input)}")
        try:
            result = handler(request)
            status = _result_status(result)
        except Exception as exc:
            status = "error"
            if tracker is not None:
                tracker.finish(activity_id, status="error", error=str(exc))
            if journal is not None:
                journal.append("tool.finished", {
                    "run_id": run_id,
                    "tool_name": tool_name,
                    "output": str(exc),
                    "status": status,
                })
            raise
        if tracker is not None:
            tracker.finish(activity_id, status=status, output_size=len(_tool_text(result)))
        if journal is not None:
            journal.append("tool.finished", {
                "run_id": run_id,
                "tool_name": tool_name,
                "output": _tool_text(result)[:2000],
                "status": status,
            })
        return result

    @override
    async def awrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        runtime = getattr(request, "runtime", None)
        journal = self._journal(runtime) if runtime is not None else None
        run_id = self._run_id(runtime) if runtime is not None else None
        tool_call = getattr(request, "tool_call", None) or {}
        tool_name = tool_call.get("name", "") if isinstance(tool_call, dict) else ""
        tool_input = tool_call.get("args", {}) if isinstance(tool_call, dict) else {}

        if journal is not None:
            journal.append("tool.called", {
                "run_id": run_id,
                "tool_name": tool_name,
                "tool_input": tool_input,
            })
        tracker = self._activity_tracker(runtime) if runtime is not None else None
        activity_id = f"{tool_name}:{tool_call.get('id', '')}" if isinstance(tool_call, dict) else tool_name
        if tracker is not None:
            tracker.start(activity_id, "tool", f"{tool_name}: {_truncate_tool_input(tool_input)}")
        try:
            result = await handler(request)
            status = _result_status(result)
        except Exception as exc:
            status = "error"
            if tracker is not None:
                tracker.finish(activity_id, status="error", error=str(exc))
            if journal is not None:
                journal.append("tool.finished", {
                    "run_id": run_id,
                    "tool_name": tool_name,
                    "output": str(exc),
                    "status": status,
                })
            raise
        if tracker is not None:
            tracker.finish(activity_id, status=status, output_size=len(_tool_text(result)))
        if journal is not None:
            journal.append("tool.finished", {
                "run_id": run_id,
                "tool_name": tool_name,
                "output": _tool_text(result)[:2000],
                "status": status,
            })
        return result
