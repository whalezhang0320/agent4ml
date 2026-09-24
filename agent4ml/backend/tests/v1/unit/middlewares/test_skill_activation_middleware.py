"""SkillActivationMiddleware 单测（F3）。

验证：
- test_before_model_matches_keyword: user message 含 "frontend" → 写 skill_suggestion
- test_before_model_no_match_returns_none: 无关键词匹配 → 返 None
- test_before_model_no_messages_returns_none: 空 messages → 返 None
- test_before_model_non_string_content: content 非 str → 返 None
- test_before_model_zero_llm_calls: 纯关键词匹配，零 LLM 调用
- test_match_keywords_dedupes: 同 skill 被多关键词匹配时去重
- test_writes_bypass_state_not_system_prompt: 写 skill_suggestion，不修改 system prompt
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent4ml.backend.agents.middlewares.skill_activation_middleware import (
    SkillActivationMiddleware,
)


def _msg(content: str) -> SimpleNamespace:
    return SimpleNamespace(content=content)


def _runtime() -> MagicMock:
    return MagicMock()


def test_before_model_matches_keyword():
    """user message 含 "frontend" → 写 skill_suggestion。"""
    mw = SkillActivationMiddleware()
    state = {"messages": [_msg("help me build a frontend UI")]}
    result = mw.before_model(state, _runtime())
    assert result is not None
    assert "skill_suggestion" in result
    suggestions = result["skill_suggestion"]
    assert isinstance(suggestions, list)
    assert len(suggestions) >= 1
    # 至少含 frontend-design
    skills = [s["skill"] for s in suggestions]
    assert "frontend-design" in skills
    # 含 keyword 字段
    assert suggestions[0]["keyword"] == "frontend"


def test_before_model_no_match_returns_none():
    """无关键词匹配 → 返 None。"""
    mw = SkillActivationMiddleware()
    state = {"messages": [_msg("what is the weather today")]}
    result = mw.before_model(state, _runtime())
    assert result is None


def test_before_model_no_messages_returns_none():
    """空 messages → 返 None。"""
    mw = SkillActivationMiddleware()
    state = {"messages": []}
    result = mw.before_model(state, _runtime())
    assert result is None


def test_before_model_missing_messages_key_returns_none():
    """state 无 messages key → 返 None。"""
    mw = SkillActivationMiddleware()
    state = {}
    result = mw.before_model(state, _runtime())
    assert result is None


def test_before_model_non_string_content():
    """content 非 str（如 list）→ 返 None。"""
    mw = SkillActivationMiddleware()
    state = {"messages": [SimpleNamespace(content=[{"type": "text", "text": "frontend"}])]}
    result = mw.before_model(state, _runtime())
    assert result is None


def test_before_model_zero_llm_calls():
    """纯关键词匹配，零 LLM 调用（INVARIANT §7.2.3）。

    验证：before_model 不调任何 LLM（无 mock_llm 参数，无 invoke 调用）。
    """
    mw = SkillActivationMiddleware()
    state = {"messages": [_msg("debug this bug")]}
    # 如果调 LLM 会抛 AttributeError（middleware 无 llm 属性）
    result = mw.before_model(state, _runtime())
    assert result is not None  # 正常返回，证明没调 LLM


def test_match_keywords_dedupes():
    """同 skill 被多关键词匹配时去重（如 "debug" + "bug" 都匹配 systematic-debugging）。"""
    mw = SkillActivationMiddleware()
    # "debug this bug" 含 debug + bug 两个关键词，都匹配 systematic-debugging
    suggestions = mw._match_keywords("debug this bug")
    skills = [s["skill"] for s in suggestions]
    # systematic-debugging 只出现一次（去重）
    assert skills.count("systematic-debugging") == 1


def test_match_keywords_multiple_skills():
    """多关键词匹配多个 skill（如 "frontend chart" → frontend-design + chart-visualization）。"""
    mw = SkillActivationMiddleware()
    suggestions = mw._match_keywords("build a frontend chart")
    skills = [s["skill"] for s in suggestions]
    assert "frontend-design" in skills
    assert "chart-visualization" in skills


def test_writes_bypass_state_not_system_prompt():
    """写 skill_suggestion 旁路存档，不修改 system prompt（INVARIANT §7.2.1）。

    验证：before_model 返 dict 只含 skill_suggestion key，不含 system prompt 修改。
    """
    mw = SkillActivationMiddleware()
    state = {"messages": [_msg("tdd workflow")]}
    result = mw.before_model(state, _runtime())
    assert result is not None
    # 只含 skill_suggestion key，不含 system_prompt / messages 等修改
    assert list(result.keys()) == ["skill_suggestion"]


def test_multiple_keywords_for_github():
    """github 相关多关键词匹配多个 skill。"""
    mw = SkillActivationMiddleware()
    suggestions = mw._match_keywords("review this github pr")
    skills = [s["skill"] for s in suggestions]
    assert "github-code-review" in skills
    assert "github-pr-workflow" in skills


def test_ml_reproduction_keywords_suggest_local_research_skills():
    mw = SkillActivationMiddleware()
    suggestions = mw._match_keywords("复现机器学习论文并分析训练日志和结果分析")
    skills = [s["skill"] for s in suggestions]
    assert "ml-paper-reproduction" in skills
    assert "ml-experiment-tracking" in skills
    assert "ml-result-analysis" in skills
