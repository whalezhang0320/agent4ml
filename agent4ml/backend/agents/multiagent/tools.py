"""tools.py — 动态生成 specialist tool + subagent tool。

设计（spec.md tools.py Requirement + design.md §3）:
- make_specialist_tool(name, specialist, context_summarizer, result_summarizer) → BaseTool
- make_subagent_tool(subagent_provider, context_summarizer, result_summarizer) → BaseTool
- tool schema 精简：LLM 只填 3 参数（goal + success_criteria + sandbox_id 可选）
- tool handler 内部编排：ContextSummarizer → specialist.invoke → ResultSummarizer → 返 JSON
- specialist 失败返 error JSON（LLM 决策 retry/fallback，INV#6 pairing 完整性）
- 新增 specialist 只注册到 SpecialistRegistry，factory 自动生成 tool，零代码侵入
- ThreadState via ContextVar（OrchestrationMiddleware Batch 9 设置）
"""
from __future__ import annotations

import contextvars
import json
from typing import Any

from langchain_core.tools import BaseTool, tool

from agent4ml.backend.agents.multiagent.context_summarizer import ContextSummarizer
from agent4ml.backend.agents.multiagent.exceptions import (
    SpecialistError,
    SubagentError,
)
from agent4ml.backend.agents.multiagent.result_summarizer import ResultSummarizer
from agent4ml.backend.agents.multiagent.specialist import SpecialistAgent
from agent4ml.backend.agents.multiagent.subagent import SubagentProvider
from agent4ml.backend.agents.multiagent.types import (
    SpecialistRequest,
    SubagentRequest,
)

_current_state: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "multiagent_state", default=None,
)


def set_current_state(state: dict) -> None:
    """OrchestrationMiddleware 调用：设置当前 ThreadState 供 tool handler 读取。"""
    _current_state.set(state)


def get_current_state() -> dict:
    """tool handler 调用：获取当前 ThreadState（未设置时返空 dict）。"""
    return _current_state.get() or {}


def _extract_sandbox_id(state: dict, sandbox_id: str | None) -> str | None:
    if sandbox_id is not None:
        return sandbox_id
    sandbox = state.get("sandbox")
    if isinstance(sandbox, dict):
        return sandbox.get("sandbox_id")
    return None


def _write_decision_log_async(
    writer: Any, specialist_name: str, goal: str, success_criteria: str, result: Any,
) -> None:
    """异步写 DecisionLogRecord（L3 扩展，fire-and-forget）.

    lazy import L3 类型避免循环依赖（同 L1 metrics.py pattern）.
    failure_category 从 result 获取（L2 ResultSummarizer 输出，可能为 None）.
    """
    import uuid
    from agent4ml.backend.agents.journal.events import utc_now_iso
    from agent4ml.backend.agents.multiagent.eval.types import DecisionLogRecord
    from agent4ml.backend.agents.multiagent.evolution.types import FailureCategory

    failure_category = getattr(result, "failure_category", None)
    if isinstance(failure_category, str):
        try:
            failure_category = FailureCategory(failure_category)
        except ValueError:
            failure_category = None

    record = DecisionLogRecord(
        log_id=str(uuid.uuid4()),
        specialist_name=specialist_name,
        task_id=str(uuid.uuid4()),
        goal=goal,
        success_criteria=success_criteria,
        failure_category=failure_category,
        success_criteria_met=1 if getattr(result, "success", False) else 0,
        lesson_text=None,  # MVP 不生成 lesson，L3 后置分析留 follow-up
        timestamp=utc_now_iso(),
    )
    writer.write_async(record)


def make_specialist_tool(
    name: str,
    specialist: SpecialistAgent,
    context_summarizer: ContextSummarizer,
    result_summarizer: ResultSummarizer,
    *,
    max_steps: int = 50,
    timeout_seconds: int = 600,
    version_dag: Any | None = None,
    budget_guard: Any | None = None,
    decision_log_writer: Any | None = None,
) -> BaseTool:
    """Factory：为 specialist 动态生成 delegate_to_<name> tool。

    tool handler 内部编排：ContextSummarizer → specialist.invoke → ResultSummarizer → JSON。
    LLM 只填 3 参数（goal + success_criteria + sandbox_id 可选），其余内部处理。
    L2 扩展：version_dag 非 None 时读 is_active SkillInjectionTemplate；
    budget_guard 非 None 时 check_and_record，超限返 BudgetExceeded JSON。
    L3 扩展：decision_log_writer 非 None 时异步写 DecisionLogRecord（fire-and-forget）。
    """

    @tool(f"delegate_to_{name}")
    def delegate_tool(
        goal: str,
        success_criteria: str,
        sandbox_id: str | None = None,
    ) -> str:
        """Delegate task to specialist. Provide goal and success_criteria. sandbox_id optional (uses thread sandbox if omitted)."""
        state = get_current_state()
        resolved_sandbox_id = _extract_sandbox_id(state, sandbox_id)

        # L2 BudgetGuard check (budget_guard 非 None 时，超限返 BudgetExceeded JSON)
        if budget_guard is not None:
            from types import SimpleNamespace
            cost = SimpleNamespace(tokens=0, cost_usd=0.0, calls=1)
            budget_result = budget_guard.check_and_record(name, cost)
            if not budget_result.allowed:
                return json.dumps({
                    "success": False,
                    "error": {
                        "type": "BudgetExceeded",
                        "message": f"{name} budget exceeded: {budget_result.reason}",
                        "remaining": {
                            "tokens": budget_result.remaining.tokens if budget_result.remaining else 0,
                            "cost_usd": budget_result.remaining.cost_usd if budget_result.remaining else 0.0,
                            "calls": budget_result.remaining.calls if budget_result.remaining else 0,
                        },
                        "fallback_target": "lead",
                    },
                    "suggestion": f"{name} daily budget exceeded, lead agent should execute task directly or wait UTC 0 reset.",
                })

        context_summary = context_summarizer.summarize(state, goal, success_criteria)

        request = SpecialistRequest(
            goal=goal,
            success_criteria=success_criteria,
            context_summary=context_summary,
            sandbox_id=resolved_sandbox_id,
            artifacts_path=state.get("metadata", {}).get("artifacts_path"),
            max_steps=max_steps,
            timeout_seconds=timeout_seconds,
        )

        try:
            raw = specialist.invoke(request)
        except SpecialistError as e:
            return json.dumps({
                "success": False,
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                },
                "suggestion": "retry, fallback to another specialist, or self-do",
            })

        result = result_summarizer.summarize(
            raw.raw_output,
            list(raw.artifacts),
            goal,
            success_criteria,
        )

        # L3 扩展：异步写 DecisionLogRecord（fire-and-forget，不阻塞 L1 turn）
        if decision_log_writer is not None:
            _write_decision_log_async(decision_log_writer, name, goal, success_criteria, result)

        return json.dumps({
            "success": result.success,
            "summary": result.summary,
            "specialist": result.specialist_name,
            "gap_analysis": result.gap_analysis,
            "artifacts": [
                {"path": a.path, "type": a.artifact_type}
                for a in result.artifacts
            ],
        })

    return delegate_tool


def make_subagent_tool(
    subagent_provider: SubagentProvider,
    context_summarizer: ContextSummarizer,
    result_summarizer: ResultSummarizer,
    *,
    max_steps: int = 20,
    timeout_seconds: int = 300,
) -> BaseTool:
    """Factory：生成 delegate_to_subagent tool（Agent4ML self-copy subagent）。

    leaf role（INV#4）：子 agent tool_groups 不含 multiagent，不能 spawn。
    shared thread sandbox（INV#3）：复用父 sandbox_id。
    """

    @tool("delegate_to_subagent")
    def delegate_tool(
        goal: str,
        success_criteria: str,
        sandbox_id: str | None = None,
    ) -> str:
        """Delegate task to an Agent4ML self-copy subagent (leaf role, isolated context, shared sandbox). Provide goal and success_criteria."""
        state = get_current_state()
        resolved_sandbox_id = _extract_sandbox_id(state, sandbox_id)

        context_summary = context_summarizer.summarize(state, goal, success_criteria)

        request = SubagentRequest(
            goal=goal,
            success_criteria=success_criteria,
            context_summary=context_summary,
            sandbox_id=resolved_sandbox_id,
            artifacts_path=state.get("metadata", {}).get("artifacts_path"),
            max_steps=max_steps,
            timeout_seconds=timeout_seconds,
        )

        try:
            sub_result = subagent_provider.spawn(request)
        except SubagentError as e:
            return json.dumps({
                "success": False,
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                },
                "suggestion": "retry, fallback to specialist, or self-do",
            })

        evaluated = result_summarizer.summarize(
            sub_result.summary,
            list(sub_result.artifacts),
            goal,
            success_criteria,
        )

        return json.dumps({
            "success": evaluated.success,
            "summary": evaluated.summary,
            "specialist": "subagent",
            "gap_analysis": evaluated.gap_analysis,
            "artifacts": [
                {"path": a.path, "type": a.artifact_type}
                for a in evaluated.artifacts
            ],
        })

    return delegate_tool
