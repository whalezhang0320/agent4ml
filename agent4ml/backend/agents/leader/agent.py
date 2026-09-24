from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from agent4ml.backend.agents.capabilities.registry import CapabilityRegistry


def _last_ai_message(state: Any) -> str:
    """从 state.messages 取最后一条 AIMessage 的文本内容（default 模式输出）。"""
    messages = state.get("messages", []) if isinstance(state, dict) else getattr(state, "messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            content = msg.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [
                    item["text"] if isinstance(item, dict) and "text" in item else str(item)
                    for item in content
                    if item
                ]
                return "".join(parts)
    return ""


def _resolve_actual_model_name(registry: CapabilityRegistry) -> str:
    """取实际 researcher 模型的路由 provider 名（FallbackChatModel.provider_names）。

    优于 config 静态 researcher_model（可能是 fake-researcher），反映真实路由链。
    FallbackChatModel 的 _identifying_params 含 providers + active，取 provider_names。
    """
    try:
        model = registry.get_model("researcher")
        # FallbackChatModel 有 provider_names
        names = getattr(model, "provider_names", None)
        if names:
            return ",".join(names)
        # 普通模型无 provider_names，回退 identifying_params 或 model 名
        params = getattr(model, "_identifying_params", None)
        if callable(params):
            params = params()
        if isinstance(params, dict) and params.get("model"):
            return str(params["model"])
        return getattr(model, "model", "") or str(type(model).__name__)
    except Exception:
        return "unknown"
from agent4ml.backend.agents.state.thread_state import create_initial_thread_state


@dataclass(frozen=True)
class AgentRunResult:
    run_id: str
    thread_id: str
    final_report: str
    events_path: str
    artifact_path: str | None
    state: dict[str, Any]


@dataclass
class LeaderAgent:
    """Thin shell: invokes compiled graph + collects report + saves artifact.

    ReAct intelligence (multi-turn decisions, tool calls, exit logic) lives
    inside the graph via create_agent + AgentMiddleware. This class only does
    graph.ainvoke() + reporter + artifact + outer config wiring. All logging
    is handled by RunJournalMiddleware inside the graph.
    """

    graph: Any
    capability_registry: CapabilityRegistry

    def run(self, question: str, run_context: Any) -> AgentRunResult:
        initial = create_initial_thread_state(question)
        initial["research_question"] = question
        initial["metadata"] = {"expert_mode": run_context.config.runtime.expert_mode}

        config = {
            "configurable": {
                "expert_mode": run_context.config.runtime.expert_mode,
                "run_id": run_context.run_id,
                "thread_id": run_context.thread_id,
                "journal": run_context.journal,
                "output_dir": str(run_context.output_dir),
                "plan_enabled": run_context.config.runtime.plan_enabled,
                "timezone": run_context.config.runtime.timezone,
                # 取实际模型的路由 provider 名（FallbackChatModel 的 provider_names），非 config 静态值
                "model": _resolve_actual_model_name(self.capability_registry),
            },
            # recursion_limit 从 config 推导：max_loop_steps * graph_node_multiplier。
            # 不再硬编码——让长程任务有足够图节点预算，安全网是 StallDetectionMiddleware。
            "recursion_limit": run_context.config.runtime.max_loop_steps * run_context.config.runtime.graph_node_multiplier,
        }

        # MCP tools are async-only StructuredTool; graph must run in async mode.
        final_state = asyncio.run(self.graph.ainvoke(
            {
                "messages": [HumanMessage(content=question)],
                "user_input": question,
                "research_question": question,
            },
            config=config,
        ))

        expert_mode = run_context.config.runtime.expert_mode
        artifact_path = None

        if expert_mode:
            # expert 模式：ReportMiddleware after_agent 已写 final_report 字段；用此 + 保存 artifact
            final_report = final_state.get("final_report") if isinstance(final_state, dict) else None
            if not final_report:
                # fallback：ReportMiddleware 未跑或 observations 空 → reporter 合成
                report_result = self.capability_registry.get_reporter().generate_report(final_state, run_context)
                final_report = report_result.final_report
            if run_context.config.reporting.save_artifact:
                artifact = self.capability_registry.get_artifact_store().save_artifact(
                    content=final_report,
                    output_dir=run_context.output_dir,
                    title="Final Report",
                    filename="final_report.md",
                    metadata={"mode": "expert"},
                )
                artifact_path = artifact.path
                run_context.journal.append(
                    "report.generated",
                    {
                        "artifact_id": artifact.artifact_id,
                        "title": artifact.title,
                        "path": artifact.path,
                        "mode": "expert",
                    },
                )
        else:
            # default 模式：不自动报告，输出 last AIMessage；不保存 artifact（靠 /report 触发）
            final_report = _last_ai_message(final_state) or ""

        return AgentRunResult(
            run_id=run_context.run_id,
            thread_id=run_context.thread_id,
            final_report=final_report,
            events_path=str(run_context.events_path),
            artifact_path=artifact_path,
            state=final_state if isinstance(final_state, dict) else dict(final_state),
        )
