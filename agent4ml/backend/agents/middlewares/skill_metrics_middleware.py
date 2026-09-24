"""SkillMetricsMiddleware — applied 打点 + completions/fallbacks 归因。

INVARIANT:
- awrap_tool_call: tool ∈ active skill.allowed_tools → _applied_ctx[sid]=True + journal skill.apply
- after_agent: 判 task_completed（run 级，与 skill 正交）+ 对 active skill 归因 record_outcome
  - tool-skill（有 allowed_tools）未命中工具 → applied=False
  - guidance-skill（无 allowed_tools）→ applied=None（不强判）
  - applied=None → 只 selections 已打，不归因 completion/fallback（INV#9）
- provenance 经 _ctx ContextVar 桥接（awrap_tool_call 无 state）
- 无 _applied_ctx（无 injection middleware）→ 降级，after_agent 跳过归因（§6.6）
- 无 store/journal 静默不报错（INV#13）
- aafter_agent 委托同步
"""
from __future__ import annotations

from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langgraph.runtime import Runtime

from agent4ml.backend.agents.middlewares.run_journal_middleware import _get_runtime_value
from agent4ml.backend.agents.skill._ctx import _active_skills_ctx, _applied_ctx


class SkillMetricsMiddleware(AgentMiddleware):
    """applied + completions/fallbacks 打点。provenance 贯穿。"""

    def __init__(self, store: Any) -> None:
        self._store = store

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        tool_call = getattr(request, "tool_call", None) or {}
        tool_name = tool_call.get("name", "") if isinstance(tool_call, dict) else ""
        runtime = getattr(request, "runtime", None)

        active_list = _active_skills_ctx.get()
        applied_map = _applied_ctx.get()
        if active_list and applied_map is not None and tool_name:
            journal = _get_runtime_value(runtime, "journal", None)
            for sid, allowed_tools in active_list:
                if tool_name in allowed_tools:
                    applied_map[sid] = True
                    if journal is not None:
                        try:
                            journal.append("skill.apply", {
                                "skill_id": sid, "tool_name": tool_name,
                            })
                        except Exception:
                            pass
            _applied_ctx.set(applied_map)

        return await handler(request)

    def after_agent(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        store = _get_runtime_value(runtime, "skill_store") or self._store
        if store is None:
            return None
        run_id = _get_runtime_value(runtime, "run_id", None)

        metadata = state.get("metadata") or {}
        active_ids = metadata.get("active_skills") or []
        if not active_ids:
            return None

        applied_map = _applied_ctx.get()
        task_completed = self._judge_task_completed(state)

        for sid in active_ids:
            try:
                rec = store.get(sid)
            except Exception:
                rec = None
            applied = applied_map.get(sid) if applied_map else None
            # tool-skill 有 allowed_tools 但未命中 → False（有机会用没用）
            if rec and rec.allowed_tools and applied is None:
                applied = False
            # guidance-skill（无 allowed_tools）→ applied 保持 None
            try:
                store.record_outcome(sid, run_id, applied, task_completed)
            except Exception:
                pass
        return None

    async def aafter_agent(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        return self.after_agent(state, runtime)

    @staticmethod
    def _judge_task_completed(state: Any) -> bool:
        """run 级任务完成判定。近似代理信号，非 ground-truth（INV#8）。

        run 模式：final_report 生成 + 无硬失败。
        chat 模式（无 report）：无硬失败（宽松）。
        """
        errors = state.get("errors") or []
        hard_failures = [e for e in errors if not _is_success_error(e)]
        if state.get("final_report"):
            return len(hard_failures) == 0
        return len(hard_failures) == 0


def _is_success_error(e: Any) -> bool:
    """errors 条目是否为 success 类（非硬失败）。兼容 dict / AgentError dataclass。"""
    if isinstance(e, dict):
        return e.get("kind") == "success"
    return getattr(e, "kind", None) == "success"
