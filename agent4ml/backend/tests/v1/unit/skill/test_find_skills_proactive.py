"""find-skills PROACTIVE + Skill-First Principle 测试（F2）。

验证：
- test_skill_first_principle_injected_when_skills_enabled: skills_enabled=True 时 prompt 含 Skill-First Principle
- test_skill_first_principle_not_injected_when_disabled: skills_enabled=False 时 prompt 不含
- test_find_skills_skill_md_has_proactive_section: SKILL.md 含 PROACTIVE CHECK 流程
- test_find_skills_skill_md_has_keyword_table: SKILL.md 含 keywords → skill categories 表
"""
from __future__ import annotations

from pathlib import Path

import pytest

from agent4ml.backend.agents.leader.prompts import apply_prompt_template


def test_skill_first_principle_injected_when_skills_enabled():
    """skills_enabled=True 时 system prompt 含 Skill-First Principle 段。"""
    prompt = apply_prompt_template(expert_mode=False, skills_enabled=True)
    assert "Skill-First Principle" in prompt
    assert "skill_search" in prompt


def test_skill_first_principle_not_injected_when_disabled():
    """skills_enabled=False 时 system prompt 不含 Skill-First Principle（保护 prompt caching）。"""
    prompt = apply_prompt_template(expert_mode=False, skills_enabled=False)
    assert "Skill-First Principle" not in prompt


def test_skill_first_principle_in_expert_mode():
    """expert_mode=True + skills_enabled=True 时 prompt 含 Skill-First Principle。"""
    prompt = apply_prompt_template(expert_mode=True, skills_enabled=True)
    assert "Skill-First Principle" in prompt


def test_skill_first_principle_and_specialist_routing_coexist():
    """skills_enabled + specialist_registry 都启用时 prompt 含两段。"""
    from unittest.mock import MagicMock

    mock_registry = MagicMock()
    mock_registry.list_specialists.return_value = ["codex"]
    prompt = apply_prompt_template(
        expert_mode=False,
        specialist_registry=mock_registry,
        skills_enabled=True,
    )
    assert "<specialist_routing>" in prompt
    assert "Skill-First Principle" in prompt


def test_find_skills_skill_md_has_proactive_section():
    """find-skills SKILL.md 含 PROACTIVE CHECK 流程。"""
    skill_md = (
        Path(__file__).parent.parent.parent.parent.parent
        / "agents"
        / "skill"
        / "builtin_skills"
        / "core"
        / "find-skills"
        / "SKILL.md"
    )
    content = skill_md.read_text(encoding="utf-8")
    assert "PROACTIVE" in content
    assert "PROACTIVE CHECK" in content
    assert "skill_search" in content


def test_find_skills_skill_md_has_keyword_table():
    """find-skills SKILL.md 含 keywords → skill categories 表。"""
    skill_md = (
        Path(__file__).parent.parent.parent.parent.parent
        / "agents"
        / "skill"
        / "builtin_skills"
        / "core"
        / "find-skills"
        / "SKILL.md"
    )
    content = skill_md.read_text(encoding="utf-8")
    assert "frontend" in content
    assert "frontend-design" in content
    assert "chart-visualization" in content
    assert "systematic-debugging" in content
    assert "test-driven-development" in content
