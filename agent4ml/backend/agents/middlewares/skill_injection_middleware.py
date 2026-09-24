"""SkillInjectionMiddleware — before_model 注入 active skills + selections 打点 + provenance。

INVARIANT:
- 混合注入：用户 override（metadata.skill_override）+ agent 自动 select（Selector）
- before_model 注入 SystemMessage + record_selection + journal skill.select
- provenance 锚点：metadata.active_skills + metadata.skill_applied（{id: None}，awrap_tool_call 标 True）
- 无 active 返 None
- 无 store/journal 静默不报错（INV#13）
- abefore_model 委托同步
"""
from __future__ import annotations

from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import SystemMessage
from langgraph.runtime import Runtime

from agent4ml.backend.agents.middlewares.run_journal_middleware import _get_runtime_value
from agent4ml.backend.agents.skill._ctx import _active_skills_ctx, _applied_ctx
from agent4ml.backend.agents.skill.injector import build_injection_text


class SkillInjectionMiddleware(AgentMiddleware):
    """before_model 注入 active skills 内容 + selections 打点 + provenance。"""

    def __init__(self, store: Any, selector: Any) -> None:
        self._store = store
        self._selector = selector

    def before_model(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        store = _get_runtime_value(runtime, "skill_store") or self._store
        journal = _get_runtime_value(runtime, "journal", None)
        run_id = _get_runtime_value(runtime, "run_id", None)

        user_input = state.get("user_input", "") or ""
        # override：configurable（/skill 命令经主循环注入）优先，state.metadata 程序注入兜底
        overrides = _get_runtime_value(runtime, "skill_override")
        if not overrides:
            overrides = (state.get("metadata") or {}).get("skill_override") or []

        active = self._selector.select_for_task(user_input, overrides=overrides)
        if not active:
            return None

        ids: list[str] = []
        for rec in active:
            try:
                store.record_selection(rec.skill_id)
            except Exception:
                pass
            if journal is not None:
                try:
                    journal.append("skill.select", {
                        "skill_id": rec.skill_id,
                        "name": rec.name,
                        "run_id": run_id,
                    })
                except Exception:
                    pass
            ids.append(rec.skill_id)

        injection = build_injection_text(active)
        try:
            from langgraph.config import get_stream_writer
            get_stream_writer()({
                "type": "skill_active",
                "skills": [rec.name for rec in active],
            })
        except Exception:
            pass
        # provenance ContextVar：供 SkillMetricsMiddleware.awrap_tool_call 读 allowed_tools + 标 applied
        _active_skills_ctx.set([(rec.skill_id, rec.allowed_tools) for rec in active])
        _applied_ctx.set({sid: None for sid in ids})
        return {
            "messages": [SystemMessage(content=injection)],
            "metadata": {
                "active_skills": ids,
                "skill_names": [rec.name for rec in active],
                "skill_applied": {sid: None for sid in ids},
            },
        }

    async def abefore_model(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        return self.before_model(state, runtime)
