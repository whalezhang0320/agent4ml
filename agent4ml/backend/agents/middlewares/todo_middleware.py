"""TodoMiddleware — extends TodoListMiddleware with context-loss detection and Nag reminder.

Three-layer protection:
1. before_model: context-loss detection (write_todos truncated → inject reminder)
2. after_model: completion enforcement (todos incomplete + LLM wants to exit → jump_to model, max 2 times)
3. before_model: Nag — dual-threshold reminder.
   - steps_since_write >= 5 guards "forgot": LLM buried in work, hasn't touched todos in a while.
   - steps_since_reminder >= 5 guards "spam": don't re-nag every step while LLM is mid-subtask.
   Both must trip (AND) to fire; firing resets only steps_since_reminder.
"""

from __future__ import annotations

import threading
from collections.abc import Awaitable, Callable
from typing import Any, override

from langchain.agents.middleware import TodoListMiddleware
from langchain.agents.middleware.todo import Todo
from langchain.agents.middleware.types import ModelRequest, ModelResponse, hook_config
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from agent4ml.backend.agents.middlewares.run_journal_middleware import _get_runtime_value
from agent4ml.backend.agents.middlewares import _jump_budget
from agent4ml.backend.agents.prompts import get_prompt_manager
from agent4ml.backend.agents.state.types import ThreadState

_STEPS_SINCE_WRITE_THRESHOLD = 5
_STEPS_SINCE_REMINDER_THRESHOLD = 5
_NAG_MIN_TODOS = 3
_MAX_COMPLETION_REMINDERS = 2


def _todos_in_messages(messages: list[Any]) -> bool:
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.get("name") == "write_todos":
                    return True
    return False


def _count_write_todos(messages: list[Any]) -> int:
    count = 0
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.get("name") == "write_todos":
                    count += 1
    return count


def _reminder_in_messages(messages: list[Any]) -> bool:
    for msg in messages:
        if isinstance(msg, HumanMessage) and getattr(msg, "name", None) == "todo_reminder":
            return True
    return False


def _format_todos(todos: list[Todo]) -> str:
    return "\n".join(f"- [{t.get('status', 'pending')}] {t.get('content', '')}" for t in todos)


def _format_completion_reminder(todos: list[Todo]) -> str:
    incomplete = [t for t in todos if t.get("status") != "completed"]
    lines = "\n".join(f"- [{t.get('status', 'pending')}] {t.get('content', '')}" for t in incomplete)
    return get_prompt_manager().load("todo", "completion_reminder", lines=lines)


def _infer_current_step_id(last_ai: AIMessage | None) -> str | None:
    """从最新 AIMessage 的 write_todos tool_call 解析 in_progress 项，派生 step_id = f"todo-{index}"。"""
    if not last_ai or not getattr(last_ai, "tool_calls", None):
        return None
    for tc in last_ai.tool_calls:
        if tc.get("name") == "write_todos":
            todos = (tc.get("args") or {}).get("todos", []) or []
            for idx, t in enumerate(todos):
                if t.get("status") == "in_progress":
                    return f"todo-{idx}"
            return None
    return None


def _has_tool_call_intent(message: AIMessage) -> bool:
    if message.tool_calls:
        return True
    if getattr(message, "invalid_tool_calls", None):
        return True
    additional_kwargs = getattr(message, "additional_kwargs", {}) or {}
    if additional_kwargs.get("tool_calls") or additional_kwargs.get("function_call"):
        return True
    response_metadata = getattr(message, "response_metadata", {}) or {}
    return response_metadata.get("finish_reason") in {"tool_calls", "function_call"}


def _has_persistent_failures(state: Any) -> bool:
    """F8.4：判断 errors 是否有 tool 持续失败超阈（attempt≥3 被禁）。

    从 state.errors 派生：任一 tool 最新条目 attempt≥3 → 持续失败，放行退出。
    """
    if not isinstance(state, dict):
        return False
    errors = state.get("errors") or []
    latest: dict[str, int] = {}
    for err in errors:
        tn = _err_field(err, "tool_name")
        att = _err_field(err, "attempt")
        if tn and att is not None:
            latest[tn] = int(att)
    return any(att >= 3 for att in latest.values())


def _err_field(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


class TodoMiddleware(TodoListMiddleware):
    """Extends TodoListMiddleware with context-loss detection and completion enforcement.

    Inherits write_todos tool registration and system-prompt injection from base class.
    """

    state_schema = ThreadState  # type: ignore[assignment]

    def __init__(self, *args: Any, enforce_completion: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._enforce_completion = enforce_completion
        self._lock = threading.Lock()
        self._pending_completion_reminders: dict[tuple[str, str], list[str]] = {}
        self._completion_reminder_counts: dict[tuple[str, str], int] = {}
        self._steps_since_write: dict[str, int] = {}
        self._steps_since_reminder: dict[str, int] = {}
        self._last_write_count: dict[str, int] = {}

    @staticmethod
    def _get_thread_id(runtime: Runtime) -> str:
        tid = _get_runtime_value(runtime, "thread_id", None)
        return str(tid) if tid else "default"

    @staticmethod
    def _get_run_id(runtime: Runtime) -> str:
        rid = _get_runtime_value(runtime, "run_id", None)
        return str(rid) if rid else "default"

    def _pending_key(self, runtime: Runtime) -> tuple[str, str]:
        return self._get_thread_id(runtime), self._get_run_id(runtime)

    @override
    def before_model(self, state: ThreadState, runtime: Runtime) -> dict[str, Any] | None:
        todos: list[Todo] = state.get("todos") or []  # type: ignore[assignment]
        messages = state.get("messages") or []

        # Layer 1: context-loss detection
        if todos and not _todos_in_messages(messages) and not _reminder_in_messages(messages):
            return {"messages": [HumanMessage(
                name="todo_reminder",
                additional_kwargs={"hide_from_ui": True},
                content=get_prompt_manager().load("todo", "context_loss_reminder", todos=_format_todos(todos)),
            )]}

        # Layer 3: Nag — dual-threshold reminder.
        thread_id = self._get_thread_id(runtime)
        incomplete = [t for t in todos if t.get("status") != "completed"]
        write_count = _count_write_todos(messages)
        with self._lock:
            if write_count > self._last_write_count.get(thread_id, 0):
                self._last_write_count[thread_id] = write_count
                self._steps_since_write[thread_id] = 0
            else:
                self._steps_since_write[thread_id] = self._steps_since_write.get(thread_id, 0) + 1
            self._steps_since_reminder[thread_id] = self._steps_since_reminder.get(thread_id, 0) + 1

            should_nag = (
                bool(incomplete)
                and len(todos) >= _NAG_MIN_TODOS
                and self._steps_since_write[thread_id] >= _STEPS_SINCE_WRITE_THRESHOLD
                and self._steps_since_reminder[thread_id] >= _STEPS_SINCE_REMINDER_THRESHOLD
            )
            if should_nag:
                self._steps_since_reminder[thread_id] = 0
                steps = self._steps_since_write[thread_id]

        if should_nag:
            return {"messages": [HumanMessage(
                name="todo_nag",
                additional_kwargs={"hide_from_ui": True},
                content=get_prompt_manager().load("todo", "nag_reminder", steps=steps, lines=_format_todos(incomplete)),
            )]}

        return None

    @override
    async def abefore_model(self, state: ThreadState, runtime: Runtime) -> dict[str, Any] | None:
        return self.before_model(state, runtime)

    # ------------------------------------------------------------------ #
    # Layer 2: completion enforcement                                       #
    # ------------------------------------------------------------------ #

    def _queue_completion_reminder(self, runtime: Runtime, reminder: str) -> None:
        key = self._pending_key(runtime)
        with self._lock:
            self._pending_completion_reminders.setdefault(key, []).append(reminder)
            self._completion_reminder_counts[key] = self._completion_reminder_counts.get(key, 0) + 1

    def _drain_completion_reminders(self, runtime: Runtime) -> list[str]:
        key = self._pending_key(runtime)
        with self._lock:
            return self._pending_completion_reminders.pop(key, [])

    def _completion_reminder_count(self, runtime: Runtime) -> int:
        key = self._pending_key(runtime)
        with self._lock:
            return self._completion_reminder_counts.get(key, 0)

    def _clear_run_state(self, runtime: Runtime) -> None:
        key = self._pending_key(runtime)
        with self._lock:
            self._pending_completion_reminders.pop(key, None)
            self._completion_reminder_counts.pop(key, None)

    def _clear_other_runs(self, runtime: Runtime) -> None:
        thread_id, current_run_id = self._pending_key(runtime)
        with self._lock:
            stale = [k for k in list(self._pending_completion_reminders) if k[0] == thread_id and k[1] != current_run_id]
            for k in stale:
                self._pending_completion_reminders.pop(k, None)
                self._completion_reminder_counts.pop(k, None)

    def _reset_nag_counters(self, runtime: Runtime) -> None:
        thread_id = self._get_thread_id(runtime)
        with self._lock:
            self._steps_since_write.pop(thread_id, None)
            self._steps_since_reminder.pop(thread_id, None)
            self._last_write_count.pop(thread_id, None)

    @hook_config(can_jump_to=["model"])
    @override
    def after_model(self, state: ThreadState, runtime: Runtime) -> dict[str, Any] | None:
        # 1. Preserve base class logic (parallel write_todos detection).
        base_result = super().after_model(state, runtime)
        if base_result is not None:
            return base_result

        messages = state.get("messages") or []
        last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)

        # 2. 推断 current_step_id（从 write_todos 调用解析 in_progress 项，D10）。
        step_id = _infer_current_step_id(last_ai)
        step_update: dict[str, Any] = {"current_step_id": step_id} if step_id is not None else {}

        # 3. Only intercept clean final answers (no tool-call intent).
        if not last_ai or _has_tool_call_intent(last_ai):
            return step_update or None

        # default 模式（enforce_completion=False）：不强制完成度，模型想退就退。
        # 软引导（prompt + write_todos 工具注册 + context-loss 检测）仍保留。
        if not self._enforce_completion:
            return step_update or None

        # 4. Allow exit when all todos are completed or none exist.
        todos: list[Todo] = state.get("todos") or []  # type: ignore[assignment]
        if not todos or all(t.get("status") == "completed" for t in todos):
            return step_update or None

        # 4b. F8.4：失败超阈放行——工具持续失败时不强制 all-completed，让任务跑到结尾产带缺口报告。
        if _has_persistent_failures(state):
            return step_update or None

        # 5. Enforce reminder cap to prevent infinite loops.
        if self._completion_reminder_count(runtime) >= _MAX_COMPLETION_REMINDERS:
            return step_update or None

        # 6. 共享 jump 预算门（与 Reflection 合计 ≤3，D6）。
        if not _jump_budget.try_consume(runtime):
            return step_update or None

        # 7. Queue reminder and jump back to model node.
        self._queue_completion_reminder(runtime, _format_completion_reminder(todos))
        return {"jump_to": "model", **step_update}

    @hook_config(can_jump_to=["model"])
    @override
    async def aafter_model(self, state: ThreadState, runtime: Runtime) -> dict[str, Any] | None:
        return self.after_model(state, runtime)

    # ------------------------------------------------------------------ #
    # wrap_model_call: inject queued completion reminders                  #
    # ------------------------------------------------------------------ #

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        reminders = self._drain_completion_reminders(request.runtime)
        if not reminders:
            return super().wrap_model_call(request, handler)
        new_messages = [
            *request.messages,
            HumanMessage(
                content="\n\n".join(dict.fromkeys(reminders)),
                name="todo_completion_reminder",
                additional_kwargs={"hide_from_ui": True},
            ),
        ]
        return super().wrap_model_call(request.override(messages=new_messages), handler)

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        reminders = self._drain_completion_reminders(request.runtime)
        if not reminders:
            return await super().awrap_model_call(request, handler)
        new_messages = [
            *request.messages,
            HumanMessage(
                content="\n\n".join(dict.fromkeys(reminders)),
                name="todo_completion_reminder",
                additional_kwargs={"hide_from_ui": True},
            ),
        ]
        return await super().awrap_model_call(request.override(messages=new_messages), handler)

    # ------------------------------------------------------------------ #
    # per-run state lifecycle                                              #
    # ------------------------------------------------------------------ #

    @override
    def before_agent(self, state: ThreadState, runtime: Runtime) -> dict[str, Any] | None:
        self._clear_other_runs(runtime)
        self._reset_nag_counters(runtime)
        return None

    @override
    async def abefore_agent(self, state: ThreadState, runtime: Runtime) -> dict[str, Any] | None:
        self._clear_other_runs(runtime)
        self._reset_nag_counters(runtime)
        return None

    @override
    def after_agent(self, state: ThreadState, runtime: Runtime) -> dict[str, Any] | None:
        self._clear_run_state(runtime)
        _jump_budget.clear(runtime)
        return None

    @override
    async def aafter_agent(self, state: ThreadState, runtime: Runtime) -> dict[str, Any] | None:
        self._clear_run_state(runtime)
        _jump_budget.clear(runtime)
        return None
