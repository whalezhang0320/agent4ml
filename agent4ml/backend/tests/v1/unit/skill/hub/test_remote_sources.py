"""WellKnownSource + ClaudeMarketplaceSource 单测（H3）。

验证：
- WellKnownSource: HTTP GET index.json + 关键词匹配 + endpoint 不可达降级
- ClaudeMarketplaceSource: 拉 registry + SkillMeta 转换 + 不可达降级
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent4ml.backend.agents.skill.hub.sources.well_known_source import WellKnownSource
from agent4ml.backend.agents.skill.hub.sources.claude_marketplace_source import (
    ClaudeMarketplaceSource,
)


# ---------------------------------------------------------------------------
# WellKnownSource
# ---------------------------------------------------------------------------


def _mock_httpx_response(data, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    return resp


def test_well_known_source_search_returns_matches():
    """well-known endpoint 返匹配 skill。"""
    mock_data = [
        {"name": "react-patterns", "description": "React best practices", "category": "frontend"},
        {"name": "vue-guide", "description": "Vue.js guide", "category": "frontend"},
    ]
    source = WellKnownSource(endpoints=["https://example.com"])
    with patch("httpx.get", return_value=_mock_httpx_response(mock_data)):
        results = source.search("react")

    assert len(results) == 1
    assert results[0].name == "react-patterns"
    assert results[0].source == "well-known"
    assert "example.com" in results[0].identifier


def test_well_known_source_search_no_match():
    """无匹配时返空。"""
    mock_data = [{"name": "other", "description": "other skill", "category": "misc"}]
    source = WellKnownSource(endpoints=["https://example.com"])
    with patch("httpx.get", return_value=_mock_httpx_response(mock_data)):
        results = source.search("frontend")
    assert results == []


def test_well_known_source_endpoint_unreachable_returns_empty():
    """endpoint 不可达时降级返空（不抛异常）。"""
    source = WellKnownSource(endpoints=["https://unreachable.example.com"])
    with patch("httpx.get", side_effect=Exception("connection refused")):
        results = source.search("anything")
    assert results == []


def test_well_known_source_http_error_returns_empty():
    """HTTP 错误（404/500）时返空。"""
    source = WellKnownSource(endpoints=["https://example.com"])
    with patch("httpx.get", return_value=_mock_httpx_response(None, status_code=404)):
        results = source.search("anything")
    assert results == []


def test_well_known_source_no_endpoints_returns_empty():
    """无 endpoints 配置时返空。"""
    source = WellKnownSource()
    assert source.search("anything") == []


def test_well_known_source_limit_truncates():
    """limit 截断。"""
    mock_data = [
        {"name": f"skill-{i}", "description": "match", "category": "test"}
        for i in range(10)
    ]
    source = WellKnownSource(endpoints=["https://example.com"])
    with patch("httpx.get", return_value=_mock_httpx_response(mock_data)):
        results = source.search("match", limit=3)
    assert len(results) == 3


def test_well_known_source_fetch_returns_dest_dir(tmp_path):
    """MVP：fetch 返 dest_dir（HTTP 下载未实现）。"""
    source = WellKnownSource()
    dest = tmp_path / "skill"
    result = source.fetch("well-known:https://example.com@skill", dest)
    assert result == dest


def test_well_known_source_preview_returns_none():
    """MVP：preview 返 None。"""
    source = WellKnownSource()
    assert source.preview("well-known:any") is None


def test_well_known_source_name():
    assert WellKnownSource().name == "well-known"


def test_well_known_source_dict_format_skills_field():
    """index.json 是 {"skills": [...]} 格式时正确解析。"""
    mock_data = {"skills": [{"name": "test", "description": "d", "category": "c"}]}
    source = WellKnownSource(endpoints=["https://example.com"])
    with patch("httpx.get", return_value=_mock_httpx_response(mock_data)):
        results = source.search("test")
    assert len(results) == 1
    assert results[0].name == "test"


# ---------------------------------------------------------------------------
# ClaudeMarketplaceSource
# ---------------------------------------------------------------------------


def test_claude_marketplace_source_search_returns_matches():
    """registry 返匹配 skill。"""
    mock_data = [
        {"name": "claude-tdd", "description": "TDD with Claude", "category": "coding"},
        {"name": "claude-review", "description": "Code review", "category": "coding"},
    ]
    source = ClaudeMarketplaceSource()
    with patch("httpx.get", return_value=_mock_httpx_response(mock_data)):
        results = source.search("tdd")

    assert len(results) == 1
    assert results[0].name == "claude-tdd"
    assert results[0].source == "claude-marketplace"


def test_claude_marketplace_source_registry_unreachable_returns_empty():
    """registry 不可达时降级返空。"""
    source = ClaudeMarketplaceSource()
    with patch("httpx.get", side_effect=Exception("connection refused")):
        results = source.search("anything")
    assert results == []


def test_claude_marketplace_source_http_error_returns_empty():
    """HTTP 错误时返空。"""
    source = ClaudeMarketplaceSource()
    with patch("httpx.get", return_value=_mock_httpx_response(None, status_code=500)):
        results = source.search("anything")
    assert results == []


def test_claude_marketplace_source_fetch_returns_dest_dir(tmp_path):
    """MVP：fetch 返 dest_dir。"""
    source = ClaudeMarketplaceSource()
    dest = tmp_path / "skill"
    result = source.fetch("claude-marketplace:test", dest)
    assert result == dest


def test_claude_marketplace_source_preview_returns_none():
    """MVP：preview 返 None。"""
    source = ClaudeMarketplaceSource()
    assert source.preview("claude-marketplace:any") is None


def test_claude_marketplace_source_name():
    assert ClaudeMarketplaceSource().name == "claude-marketplace"


def test_claude_marketplace_source_custom_registry_url():
    """自定义 registry URL。"""
    source = ClaudeMarketplaceSource(registry_url="https://custom.registry.com/skills.json")
    assert source._registry_url == "https://custom.registry.com/skills.json"
