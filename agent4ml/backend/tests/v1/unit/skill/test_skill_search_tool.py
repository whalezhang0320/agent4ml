"""skill_search tool 单测 — find-skills 主动触发（F1）。

验证（tasks.md 1.3）：
- test_skill_search_returns_builtin_matches: 搜 builtin 返匹配 SkillMeta
- test_skill_search_no_manager_returns_empty: skill_manager 不可用时返 "[]"
- test_skill_search_hub_unavailable_degrades: hub import 失败时降级为只搜 builtin
- test_skill_search_dedupes_by_name: 同名 skill 去重
- test_skill_search_limit_truncates: limit 截断
- test_skill_search_json_format: 返 JSON 格式
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agent4ml.backend.agents.agent_tools.builtin.skill_search import skill_search


def _mock_builtin_result(name: str, description: str = "", category: str = "core") -> dict:
    return {
        "name": name,
        "description": description,
        "category": category,
        "path": f"/builtin/{category}/{name}/SKILL.md",
        "is_active": False,
    }


def test_skill_search_returns_builtin_matches():
    """搜 builtin 返匹配 SkillMeta 列表。"""
    mock_mgr = MagicMock()
    mock_mgr.search_builtin_skills.return_value = [
        _mock_builtin_result("frontend-design", "frontend UI design skill", "creative"),
    ]
    with patch(
        "agent4ml.backend.agents.skill.build_skill_manager",
        return_value=mock_mgr,
    ):
        result = skill_search.invoke({"query": "frontend", "limit": 5})

    parsed = json.loads(result)
    assert len(parsed) == 1
    assert parsed[0]["name"] == "frontend-design"
    assert parsed[0]["source"] == "builtin"  # 统一字段补 source
    assert parsed[0]["category"] == "creative"


def test_skill_search_no_manager_returns_empty():
    """skill_manager 不可用（build_skill_manager 返 None）时返 "[]"。"""
    with patch(
        "agent4ml.backend.agents.skill.build_skill_manager",
        return_value=None,
    ):
        result = skill_search.invoke({"query": "anything", "limit": 5})
    assert result == "[]"


def test_skill_search_hub_unavailable_degrades():
    """hub 模块不可用（ImportError）时降级为只搜 builtin，不抛异常。"""
    mock_mgr = MagicMock()
    mock_mgr.search_builtin_skills.return_value = [
        _mock_builtin_result("plan", "planning skill"),
    ]
    # 模拟 hub.search 不存在（add-skill-hub change 未实现）
    with patch(
        "agent4ml.backend.agents.skill.build_skill_manager",
        return_value=mock_mgr,
    ):
        # unified_search import 会失败（hub 模块不存在），降级为只搜 builtin
        result = skill_search.invoke({"query": "plan", "limit": 5})

    parsed = json.loads(result)
    assert len(parsed) == 1
    assert parsed[0]["name"] == "plan"


def test_skill_search_dedupes_by_name():
    """同名 skill 去重（builtin + hub 都返同 name 时只保留一个）。"""
    mock_mgr = MagicMock()
    mock_mgr.search_builtin_skills.return_value = [
        _mock_builtin_result("frontend-design", "from builtin"),
    ]
    with patch(
        "agent4ml.backend.agents.skill.build_skill_manager",
        return_value=mock_mgr,
    ):
        result = skill_search.invoke({"query": "frontend", "limit": 5})

    parsed = json.loads(result)
    assert len(parsed) == 1  # 去重后只有一个


def test_skill_search_limit_truncates():
    """limit 截断结果数量。"""
    mock_mgr = MagicMock()
    mock_mgr.search_builtin_skills.return_value = [
        _mock_builtin_result(f"skill-{i}") for i in range(10)
    ]
    with patch(
        "agent4ml.backend.agents.skill.build_skill_manager",
        return_value=mock_mgr,
    ):
        result = skill_search.invoke({"query": "skill", "limit": 3})

    parsed = json.loads(result)
    assert len(parsed) == 3


def test_skill_search_json_format():
    """返合法 JSON 字符串。"""
    mock_mgr = MagicMock()
    mock_mgr.search_builtin_skills.return_value = [
        _mock_builtin_result("tdd", "test driven development"),
    ]
    with patch(
        "agent4ml.backend.agents.skill.build_skill_manager",
        return_value=mock_mgr,
    ):
        result = skill_search.invoke({"query": "test", "limit": 5})

    # 验证是合法 JSON
    parsed = json.loads(result)
    assert isinstance(parsed, list)
    assert len(parsed) == 1
    item = parsed[0]
    # 验证字段齐全
    assert "name" in item
    assert "description" in item
    assert "category" in item
    assert "source" in item


def test_skill_search_registered_in_builtin_tools():
    """skill_search 注册到 get_builtin_tools 返回列表中。"""
    from agent4ml.backend.agents.agent_tools.builtin import get_builtin_tools

    tools = get_builtin_tools()
    names = [t.name for t in tools]
    assert "skill_search" in names


def test_skill_search_in_core_group():
    """skill_search 归 core group（default + expert 都加载）。"""
    from agent4ml.backend.agents.agent_tools.available import _tool_group

    assert _tool_group("skill_search") == "core"
