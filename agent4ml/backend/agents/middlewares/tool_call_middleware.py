"""ToolCallMiddleware — 工具调用账本 + 失败分类 + 重试预算 + 超限禁工具 + 硬预算。

FD17-FD19：最外层 wrap_tool_call，看到所有工具最终结果 + 捕获内层异常。
成败都记 errors 账本（AgentError kind=success/failure），per-tool 连续失败 + 全局调用数从 errors 派生。
禁工具时短路 + 自发 tool.blocked 事件。仅 general/expert 启用。
"""

from __future__ import annotations

import re
import threading
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any, override

from langchain.agents.middleware.types import AgentMiddleware, hook_config
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command

from agent4ml.backend.agents.middlewares.run_journal_middleware import _get_runtime_value
from agent4ml.backend.agents.state.types import AgentError, ThreadState

_RETRY_BUDGET = 999       # per-tool 连续失败上限（禁工具）——放宽：用户要求取消限制
_HARD_BUDGET = 999        # run 级工具调用总数上限——放宽：用户要求取消限制
_SUMMARY_THRESHOLDS = (3, 6, 9)  # 失败摘要递进注入阈值

# F5：业务失败特征正则
_BLOCKED_RE = re.compile(r"blocked|forbidden|captcha|access denied|403", re.IGNORECASE)
_EMPTY_RE = re.compile(r"no results? found|no results|empty result|nothing found|0 results", re.IGNORECASE)

_REASON_MAP = {
    "network": "网络问题（超时/连接失败）",
    "rate_limit": "API 限流",
    "blocked": "内容被封锁/拒绝访问",
    "empty": "无搜索结果",
    "server_error": "服务端错误（5xx）",
    "client_error": "客户端错误（4xx，换 provider 也会失败）",
    "unknown": "未知错误",
}


def _make_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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


def _classify_exception(exc: Exception) -> str:
    """F5：按异常类型归基本错误分类。"""
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return "network"
    try:
        import openai
        if isinstance(exc, getattr(openai, "RateLimitError", type(None))):
            return "rate_limit"
        if isinstance(exc, getattr(openai, "APIStatusError", type(None))):
            status = getattr(exc, "status_code", None)
            if status is not None and 500 <= status < 600:
                return "server_error"
            if status is not None and 400 <= status < 500:
                return "client_error"
    except ImportError:
        pass
    msg = str(exc).lower()
    if "timeout" in msg or "timed out" in msg or "connection" in msg:
        return "network"
    if "rate" in msg and "limit" in msg:
        return "rate_limit"
    return "unknown"


def _classify_business_failure(text: str) -> str | None:
    """F5：HTTP 200 但内容含业务失败特征。"""
    if _BLOCKED_RE.search(text):
        return "blocked"
    if _EMPTY_RE.search(text):
        return "empty"
    return None


def _reason_for(error_type: str) -> str:
    return _REASON_MAP.get(error_type, "未知错误")


def _is_failure(result: Any) -> tuple[str, str] | None:
    """判断工具结果是否失败。返回 (error_type, reason) 或 None。

    - ToolMessage status="error" → 按内容/异常推断
    - HTTP 200 含封锁/空结果特征 → blocked/empty
    - 否则 None（成功）
    """
    if isinstance(result, ToolMessage) and getattr(result, "status", None) == "error":
        # 内层已标 error（如 Evidence 捕获异常）—— 从 content 推断分类
        text = _tool_text(result)
        for et in ("network", "rate_limit", "blocked", "empty", "server_error", "client_error"):
            if et in text.lower():
                return et, _reason_for(et)
        return "unknown", _reason_for("unknown")
    text = _tool_text(result)
    biz = _classify_business_failure(text)
    if biz:
        return biz, _reason_for(biz)
    # 检测工具返回的 error JSON（如 ddg {"error": "Search failed: ..."}）
    if '"error"' in text and ('search failed' in text.lower() or 'not installed' in text.lower() or 'failed' in text.lower()):
        return "unknown", "工具返回错误"
    return None


def _latest_attempt(errors: list, tool_name: str) -> int:
    """该 tool 最新条目的 attempt（连续失败计数，成功归 0）。"""
    for err in reversed(errors):
        if _field(err, "tool_name") == tool_name:
            return int(_field(err, "attempt") or 0)
    return 0


def _total_calls(errors: list) -> int:
    """全局工具调用数 = len(errors)（成功+失败都记）。"""
    return len(errors)


def _field(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


class ToolCallMiddleware(AgentMiddleware):
    """工具调用账本：成败都记 errors，per-tool 重试/禁工具，硬预算兜底。

    failure_summary + budget_exhausted 提示用队列延迟到 before_model 注入，
    避免在 wrap_tool_call 注入 HumanMessage 插在并行 tool_calls 的 ToolMessage 之间
    破坏 API pairing（AIMessage(tool_calls) 后必须紧跟 ToolMessage）。
    """

    state_schema = ThreadState  # type: ignore[assignment]

    def __init__(self, retry_budget: int = _RETRY_BUDGET, hard_budget: int = _HARD_BUDGET) -> None:
        self._retry_budget = retry_budget
        self._hard_budget = hard_budget
        self._lock = threading.Lock()
        self._pending_summaries: dict[tuple[str, str], list[str]] = {}
        self._run_baselines: dict[tuple[str, str], int] = {}

    def _queue_key(self, runtime: Runtime) -> tuple[str, str]:
        tid = str(_get_runtime_value(runtime, "thread_id", None) or "default")
        rid = str(_get_runtime_value(runtime, "run_id", None) or "default")
        return (tid, rid)

    def _queue_summary(self, runtime: Runtime, text: str) -> None:
        key = self._queue_key(runtime)
        with self._lock:
            self._pending_summaries.setdefault(key, []).append(text)

    def _drain_summaries(self, runtime: Runtime) -> list[str]:
        key = self._queue_key(runtime)
        with self._lock:
            return self._pending_summaries.pop(key, None) or []

    def _set_baseline(self, runtime: Runtime, count: int) -> None:
        key = self._queue_key(runtime)
        with self._lock:
            self._run_baselines[key] = count

    def _run_tool_count(self, runtime: Runtime, errors: list) -> int:
        """per-run 工具调用数 = len(errors) - baseline（baseline 在 before_agent 记录）。"""
        key = self._queue_key(runtime)
        with self._lock:
            baseline = self._run_baselines.get(key, 0)
        return max(0, len(errors) - baseline)

    def _run_errors_slice(self, runtime: Runtime, errors: list) -> list:
        """返回当前 run 的 errors 切片（baseline 之后），用于 per-run retry budget。"""
        key = self._queue_key(runtime)
        with self._lock:
            baseline = self._run_baselines.get(key, 0)
        return errors[baseline:] if baseline > 0 else errors

    def _journal(self, runtime: Runtime) -> Any:
        return _get_runtime_value(runtime, "journal", None)

    def _emit(self, runtime: Runtime, event_type: str, payload: dict) -> None:
        j = self._journal(runtime)
        if j is not None:
            j.append(event_type, payload)

    @override
    def before_agent(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        return None

    def _build_failure_summary(self, errors: list, tool_name: str) -> str:
        """F8.2：结构化失败摘要（错误类型→原因模板）。"""
        fails = [e for e in errors if _field(e, "tool_name") == tool_name and _field(e, "kind") == "failure"]
        lines = [f"工具 {tool_name} 已连续失败 {len(fails)} 次："]
        for e in fails[-3:]:
            lines.append(f"- {_field(e, 'error_type')}: {_field(e, 'reason') or _field(e, 'message')}")
        lines.append("请分析失败模式，考虑换搜索词/换工具/换方法；若判断不可恢复，基于现有证据收尾报告。")
        return "\n".join(lines)

    def _process_result(
        self, request: ToolCallRequest, result: Any, runtime: Runtime, exc: Exception | None,
    ) -> Any:
        """返回阶段处理：记 errors 账本 + 失败摘要 + 硬预算。"""
        tool_name = request.tool_call.get("name", "")
        call_id = request.tool_call.get("id", "")
        state = request.state
        errors = state.get("errors") or [] if isinstance(state, dict) else []
        # per-run errors slice 用于 retry budget 判定
        run_errors = self._run_errors_slice(runtime, errors)

        # 判定成败 + 分类
        if exc is not None:
            error_type = _classify_exception(exc)
            kind = "failure"
            attempt = _latest_attempt(run_errors, tool_name) + 1
            reason = _reason_for(error_type)
            err = AgentError(
                error_id=_make_id("err"), stage="tool",
                message=f"{tool_name}: {exc}", tool_name=tool_name,
                kind=kind, attempt=attempt, error_type=error_type, reason=reason,
                related_refs=(call_id,), created_at=_now_iso(),
            )
            self._emit(runtime, "tool.failure_streak", {"tool": tool_name, "attempt": attempt, "type": error_type})
            # 合成 error ToolMessage 补 tool_call_id，保证 tool_call/tool_response 配对完整
            # （缺则下一轮 model 调用 400: insufficient tool messages following tool_calls）
            result = ToolMessage(
                content=f"⚠️ 工具 {tool_name} 执行异常（{error_type}）：{exc}",
                tool_call_id=call_id,
                status="error",
            )
        else:
            fail = _is_failure(result)
            if fail:
                error_type, reason = fail
                kind = "failure"
                attempt = _latest_attempt(run_errors, tool_name) + 1
                err = AgentError(
                    error_id=_make_id("err"), stage="tool",
                    message=f"{tool_name}: 业务失败 {reason}", tool_name=tool_name,
                    kind=kind, attempt=attempt, error_type=error_type, reason=reason,
                    related_refs=(call_id,), created_at=_now_iso(),
                )
                self._emit(runtime, "tool.failure_streak", {"tool": tool_name, "attempt": attempt, "type": error_type})
            else:
                kind = "success"
                attempt = 0
                err = AgentError(
                    error_id=_make_id("err"), stage="tool",
                    message=f"{tool_name}: success", tool_name=tool_name,
                    kind=kind, attempt=attempt, error_type="", reason="",
                    related_refs=(call_id,), created_at=_now_iso(),
                )

        # 记账本——per-run 计数
        run_count = self._run_tool_count(runtime, errors) + 1
        update: dict[str, Any] = {"errors": [err]}

        # 守卫：result 为 None（handler 返 None 无异常）—— 补空 ToolMessage 保 tool_call 配对完整
        if result is None:
            result = ToolMessage(content="", tool_call_id=call_id)

        # F8.2：失败摘要递进注入（3/6/9）—— 队列延迟到 before_model，避免插在并行 ToolMessage 间破坏 pairing
        if kind == "failure" and attempt in _SUMMARY_THRESHOLDS:
            summary = self._build_failure_summary(errors + [err], tool_name)
            self._queue_summary(request.runtime, summary)

        # F8.5：硬预算兜底 —— per-run
        if run_count >= self._hard_budget:
            self._queue_summary(
                runtime,
                f"本轮工具调用总数已达 {run_count}（预算 {_HARD_BUDGET}），必须收尾报告。",
            )
            self._emit(runtime, "tool.budget_exhausted", {"count": run_count, "max": self._hard_budget})

        # 合并：若 result 已是 Command（内层 Evidence 返回），合并 errors 进 update
        if isinstance(result, Command):
            merged = dict(result.update)
            merged.setdefault("errors", []).extend(update["errors"])
            return Command(update=merged)
        # 非 Command（原 ToolMessage 或异常）：返 Command 带 errors + result ToolMessage
        update["messages"] = [result] if result is not None else []
        return Command(update=update)

    @hook_config(can_jump_to=["model"])
    @override
    def before_model(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        """drain 队列的 failure_summary / budget_exhausted 提示，在 before_model 注入。

        before_model 在 ToolNode 之后执行（ToolMessage 已全部就位），
        HumanMessage 注入在 ToolMessage 之后不破坏 AIMessage(tool_calls)→ToolMessage pairing。
        """
        summaries = self._drain_summaries(runtime)
        if not summaries:
            return None
        combined = "\n\n".join(dict.fromkeys(summaries))
        return {"messages": [HumanMessage(
            name="tool_failure_summary",
            additional_kwargs={"hide_from_ui": True},
            content=f"<system_reminder>\n{combined}\n</system_reminder>",
        )]}

    @hook_config(can_jump_to=["model"])
    @override
    async def abefore_model(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        return self.before_model(state, runtime)

    @override
    def before_agent(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        # 记录 errors 基线（per-run 工具调用计数基准）
        errors = state.get("errors") if isinstance(state, dict) else None
        self._set_baseline(runtime, len(errors or []))
        # 清理其他 run 的陈旧队列 + 基线
        tid = str(_get_runtime_value(runtime, "thread_id", None) or "default")
        rid = str(_get_runtime_value(runtime, "run_id", None) or "default")
        with self._lock:
            stale = [k for k in list(self._pending_summaries) if k[0] == tid and k[1] != rid]
            for k in stale:
                self._pending_summaries.pop(k, None)
                self._run_baselines.pop(k, None)
        return None

    @override
    def wrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Any],
    ) -> Any:
        tool_name = request.tool_call.get("name", "")
        state = request.state
        errors = state.get("errors") or [] if isinstance(state, dict) else []
        run_count = self._run_tool_count(request.runtime, errors)
        # per-run errors slice：只看当前 run 的 errors（baseline 后），防跨 run 持久禁工具
        run_errors = self._run_errors_slice(request.runtime, errors)

        # F8.3：禁工具短路——per-run retry budget
        if _latest_attempt(run_errors, tool_name) >= self._retry_budget:
            self._emit(request.runtime, "tool.blocked", {"tool": tool_name, "reason": "retry_budget_exhausted"})
            failure_msg = ToolMessage(
                content=f"⚠️ 工具 {tool_name} 已达重试上限（{self._retry_budget}），已被禁用，请换方法或收尾。",
                tool_call_id=request.tool_call.get("id", ""),
                status="error",
            )
            return Command(update={"messages": [failure_msg]})

        # F8.5：硬预算短路——per-run 调用数达上限，拒绝所有后续工具，强制模型收尾
        if run_count >= self._hard_budget:
            self._emit(request.runtime, "tool.budget_exhausted", {"count": run_count, "max": self._hard_budget})
            failure_msg = ToolMessage(
                content=f"⚠️ 本轮工具调用已达预算上限（{self._hard_budget}），所有工具已禁用，请立即基于现有证据输出最终报告。",
                tool_call_id=request.tool_call.get("id", ""),
                status="error",
            )
            return Command(update={"messages": [failure_msg]})

        try:
            result = handler(request)
            return self._process_result(request, result, request.runtime, exc=None)
        except Exception as exc:
            return self._process_result(request, None, request.runtime, exc=exc)

    @override
    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[Any]],
    ) -> Any:
        tool_name = request.tool_call.get("name", "")
        state = request.state
        errors = state.get("errors") or [] if isinstance(state, dict) else []
        run_count = self._run_tool_count(request.runtime, errors)
        run_errors = self._run_errors_slice(request.runtime, errors)

        if _latest_attempt(run_errors, tool_name) >= self._retry_budget:
            self._emit(request.runtime, "tool.blocked", {"tool": tool_name, "reason": "retry_budget_exhausted"})
            failure_msg = ToolMessage(
                content=f"⚠️ 工具 {tool_name} 已达重试上限（{self._retry_budget}），已被禁用，请换方法或收尾。",
                tool_call_id=request.tool_call.get("id", ""),
                status="error",
            )
            return Command(update={"messages": [failure_msg]})

        # F8.5：硬预算短路——per-run
        if run_count >= self._hard_budget:
            self._emit(request.runtime, "tool.budget_exhausted", {"count": run_count, "max": self._hard_budget})
            failure_msg = ToolMessage(
                content=f"⚠️ 本轮工具调用已达预算上限（{self._hard_budget}），所有工具已禁用，请立即基于现有证据输出最终报告。",
                tool_call_id=request.tool_call.get("id", ""),
                status="error",
            )
            return Command(update={"messages": [failure_msg]})

        try:
            result = await handler(request)
            return self._process_result(request, result, request.runtime, exc=None)
        except Exception as exc:
            return self._process_result(request, None, request.runtime, exc=exc)
