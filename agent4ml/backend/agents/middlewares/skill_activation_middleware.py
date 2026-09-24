"""SkillActivationMiddleware — before_model 主动建议相关 skill。

设计（设计文档 46 §3.2 改造点 D）：
- before_model hook 分析 user message 关键词，主动建议相关 skill
- 写 ThreadState.skill_suggestion（旁路存档，list[dict] 含 keyword + skill name）
- 不修改 system prompt（保护 prompt caching，INVARIANT §7.2.1）
- 简单关键词匹配（零 LLM 调用，INVARIANT §7.2.3）
- lead agent 看到建议后自己决定是否调 skill_search（建议非强制，INVARIANT §7.2.2）
"""
from __future__ import annotations

from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langgraph.runtime import Runtime


# 关键词 → 候选 skill name 映射（与 find-skills SKILL.md 的 keyword 表一致）
_KEYWORD_MAP: dict[str, list[str]] = {
    "frontend": ["frontend-design"],
    "ui": ["frontend-design"],
    "react": ["frontend-design"],
    "vue": ["frontend-design"],
    "chart": ["chart-visualization"],
    "graph": ["chart-visualization"],
    "visualization": ["chart-visualization"],
    "diagram": ["architecture-diagram", "concept-diagrams"],
    "github": ["github-code-review", "github-pr-workflow", "github-issues"],
    "pr": ["github-pr-workflow", "github-code-review"],
    "code review": ["github-code-review"],
    "debug": ["systematic-debugging", "python-debugpy"],
    "bug": ["systematic-debugging", "python-debugpy"],
    "test": ["test-driven-development"],
    "tdd": ["test-driven-development"],
    "plan": ["plan", "spike"],
    "spike": ["spike", "plan"],
    "refactor": ["simplify-code"],
    "simplify": ["simplify-code"],
    "machine learning": ["ml-paper-reproduction", "ml-experiment-tracking"],
    "reproduce": ["ml-paper-reproduction"],
    "reproduction": ["ml-paper-reproduction"],
    "training log": ["ml-experiment-tracking"],
    "experiment tracking": ["ml-experiment-tracking"],
    "result analysis": ["ml-result-analysis"],
    "paper results": ["ml-result-analysis"],
    "机器学习": ["ml-paper-reproduction", "ml-experiment-tracking"],
    "复现": ["ml-paper-reproduction"],
    "训练日志": ["ml-experiment-tracking"],
    "结果分析": ["ml-result-analysis"],
}


class SkillActivationMiddleware(AgentMiddleware):
    """before_model: 分析 user message 关键词，主动建议相关 skill。

    写 ThreadState.skill_suggestion（旁路存档），不修改 system prompt。
    lead agent 看到建议后自己决定是否调 skill_search / 读 SKILL.md。
    """

    def before_model(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        if not messages:
            return None

        last_msg = messages[-1]
        content = getattr(last_msg, "content", "")
        if not isinstance(content, str):
            return None

        suggestions = self._match_keywords(content.lower())
        if not suggestions:
            return None

        return {"skill_suggestion": suggestions}

    def _match_keywords(self, text: str) -> list[dict]:
        """关键词匹配 → 候选 skill name 列表。

        返 [{"keyword": "frontend", "skill": "frontend-design"}, ...] 去重。
        """
        suggestions: list[dict] = []
        seen_skills: set[str] = set()
        for keyword, skill_names in _KEYWORD_MAP.items():
            if keyword in text:
                for skill_name in skill_names:
                    if skill_name not in seen_skills:
                        seen_skills.add(skill_name)
                        suggestions.append({"keyword": keyword, "skill": skill_name})
        return suggestions
