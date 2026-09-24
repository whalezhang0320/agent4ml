"""Skill provenance ContextVar — 跨 hook 桥接 active_skills + applied 标记。

为什么需要：awrap_tool_call 无 state 参数，无法读 state.metadata。
SkillInjectionMiddleware.before_model 设此 ContextVar，
SkillMetricsMiddleware.awrap_tool_call 读/改，after_agent 读。

INVARIANT:
- _active_skills_ctx: [(skill_id, allowed_tools)] 本轮注入的 active skill
- _applied_ctx: {skill_id: bool} awrap_tool_call 标记的 applied（None=未标/guidance）
- 未设（无 injection middleware）→ 均 None，SkillMetrics 降级（§6.6）
"""
from __future__ import annotations

from contextvars import ContextVar

# [(skill_id, allowed_tools)] — 本轮 active skill 及其声明工具
_active_skills_ctx: ContextVar[list[tuple[str, tuple[str, ...]]] | None] = ContextVar(
    "agent4ml_skill_active", default=None,
)

# {skill_id: True} — awrap_tool_call 标记被实际应用的 tool-skill
_applied_ctx: ContextVar[dict[str, bool] | None] = ContextVar(
    "agent4ml_skill_applied", default=None,
)
