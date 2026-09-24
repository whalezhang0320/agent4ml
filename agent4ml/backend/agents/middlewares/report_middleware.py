"""ReportMiddleware — ReAct 循环结束后独立合成 final_report。

after_agent 阶段：observations 非空时用 reporter prompt 单独调一次 LLM，
基于 observations + sources + errors 合成结构化 Markdown 写回 state["final_report"]。
不再赌最后一条 AIMessage（deer-flow 模式）。仅 general/expert 启用。
"""

from __future__ import annotations

from typing import Any, override

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.runtime import Runtime

from agent4ml.backend.agents.prompts import get_prompt_manager
from agent4ml.backend.agents.state.types import ThreadState


def get_reporter_system() -> str:
    """加载 reporter 系统 prompt（从 prompts/system/reporter/system.md）。"""
    return get_prompt_manager().load("reporter", "system")


def _format_observations(observations: list[Any]) -> str:
    if not observations:
        return "（无观察记录）"
    lines: list[str] = []
    for obs in observations:
        oid = _field(obs, "observation_id") or "?"
        sid = _field(obs, "step_id") or "-"
        content = _field(obs, "content") or ""
        refs = _field(obs, "source_refs") or ()
        refs_str = ", ".join(refs) if refs else "无"
        lines.append(f"[{oid}] (step={sid}, sources={refs_str})\n{content}")
    return "\n\n".join(lines)


def _format_sources(sources: list[Any]) -> str:
    if not sources:
        return "（无来源）"
    lines: list[str] = []
    for src in sources:
        sid = _field(src, "source_id") or "?"
        url = _field(src, "url") or ""
        title = _field(src, "title") or ""
        lines.append(f"[{sid}] {title} — {url}".rstrip(" —"))
    return "\n".join(lines)


def _format_errors(errors: list[Any]) -> str:
    """F8.6：过滤 kind=success，只展示 failure；失败多则占比大（详细列出）。"""
    failures = [e for e in errors if _field(e, "kind") != "success"]
    if not failures:
        return ""
    lines = [f"以下工具调用失败（共 {len(failures)} 次），对应信息未能获取："]
    # 按 error_type 汇总
    by_type: dict[str, list[tuple[str, str]]] = {}  # et -> [(tool_name, reason)]
    for err in failures:
        et = _field(err, "error_type") or "unknown"
        by_type.setdefault(et, []).append((_field(err, "tool_name") or "unknown", _field(err, "reason") or ""))
    for et, entries in by_type.items():
        tools = sorted({t for t, _ in entries})
        reason = next((r for _, r in entries if r), "")
        lines.append(f"- [{et}] {reason}：涉及工具 {', '.join(tools)}")
    return "\n".join(lines)


def _field(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def _build_reporter_messages(state: dict[str, Any]) -> list[Any]:
    question = state.get("research_question") or state.get("user_input") or "Research report"
    observations = state.get("observations") or []
    sources = state.get("sources") or []
    errors = state.get("errors") or []

    user_content = (
        f"# 研究问题\n{question}\n\n"
        f"# 已收集的观察（observations）\n{_format_observations(observations)}\n\n"
        f"# 来源（sources）\n{_format_sources(sources)}\n"
    )
    err_block = _format_errors(errors)
    if err_block:
        user_content += f"\n# 工具失败（errors）\n{err_block}\n"
    user_content += "\n\n请基于以上证据撰写完整研究报告。"

    return [SystemMessage(content=get_reporter_system()), HumanMessage(content=user_content)]


class ReportMiddleware(AgentMiddleware):
    """after_agent 阶段独立合成 final_report。MVP 复用 researcher 模型（D5）。"""

    state_schema = ThreadState  # type: ignore[assignment]

    def __init__(self, model: BaseChatModel, auto_synthesize: bool = True) -> None:
        self._model = model
        self._auto_synthesize = auto_synthesize

    def _synthesize(self, state: dict[str, Any]) -> str:
        from agent4ml.backend.agents.observability.interrupt_protection import (
            interrupt_protection,
        )
        messages = _build_reporter_messages(state)
        with interrupt_protection():
            response = self._model.invoke(messages, config={"tags": ["internal_llm"]})
        return getattr(response, "content", str(response))

    async def _asynthesize(self, state: dict[str, Any]) -> str:
        from agent4ml.backend.agents.observability.interrupt_protection import (
            interrupt_protection,
        )
        messages = _build_reporter_messages(state)
        with interrupt_protection():
            response = await self._model.ainvoke(messages, config={"tags": ["internal_llm"]})
        return getattr(response, "content", str(response))

    @override
    def after_agent(self, state: ThreadState, runtime: Runtime) -> dict[str, Any] | None:
        # default 模式（auto_synthesize=False）：不自动合成，报告靠 /report 命令触发。
        if not self._auto_synthesize:
            return None
        observations = state.get("observations") or []  # type: ignore[assignment]
        if not observations:
            return None
        report = self._synthesize(state if isinstance(state, dict) else dict(state))
        return {"final_report": report}

    @override
    async def aafter_agent(self, state: ThreadState, runtime: Runtime) -> dict[str, Any] | None:
        if not self._auto_synthesize:
            return None
        observations = state.get("observations") or []  # type: ignore[assignment]
        if not observations:
            return None
        report = await self._asynthesize(state if isinstance(state, dict) else dict(state))
        return {"final_report": report}
