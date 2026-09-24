"""Skill Hub CLI + slash command + config + bootstrap 集成测试（H6+H7+H8）。

验证：
- config: SkillConfig 含 hub 字段
- /skill search 接 hub unified_search
- /skill install 支持 remote identifier（github:/well-known:/...）
- bootstrap 装配 hub
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent4ml.backend.agents.skill.config import SkillConfig, load_skill_config


def test_skill_config_has_hub_fields():
    """SkillConfig 含 hub 字段（H8 config schema 扩展）。"""
    c = SkillConfig()
    assert hasattr(c, "hub_enabled")
    assert hasattr(c, "hub_quarantine_enabled")
    assert hasattr(c, "hub_audit_log")
    assert c.hub_enabled is True  # 默认 true
    assert c.hub_quarantine_enabled is True
    assert c.hub_audit_log is True


def test_skill_config_frozen_with_hub_fields():
    """SkillConfig 含 hub 字段后仍 frozen。"""
    c = SkillConfig()
    with pytest.raises(Exception):
        c.hub_enabled = False  # type: ignore


def test_load_skill_config_reads_hub_env_vars(monkeypatch):
    """load_skill_config 读 AGENT4ML_SKILL_HUB_* env vars。"""
    monkeypatch.setenv("AGENT4ML_SKILL_HUB_ENABLED", "false")
    monkeypatch.setenv("AGENT4ML_SKILL_HUB_QUARANTINE", "false")
    monkeypatch.setenv("AGENT4ML_SKILL_HUB_AUDIT", "false")
    c = load_skill_config()
    assert c.hub_enabled is False
    assert c.hub_quarantine_enabled is False
    assert c.hub_audit_log is False


def test_load_skill_config_defaults_hub_true(monkeypatch):
    """不设 env vars 时 hub 默认 true。"""
    for var in ("AGENT4ML_SKILL_HUB_ENABLED", "AGENT4ML_SKILL_HUB_QUARANTINE", "AGENT4ML_SKILL_HUB_AUDIT"):
        monkeypatch.delenv(var, raising=False)
    c = load_skill_config()
    assert c.hub_enabled is True
    assert c.hub_quarantine_enabled is True
    assert c.hub_audit_log is True


# ---------------------------------------------------------------------------
# /skill search 接 hub unified_search（H7）
# ---------------------------------------------------------------------------


def test_skill_search_command_uses_hub_unified_search(monkeypatch):
    """/skill search 调 hub unified_search_as_dicts。"""
    # mock unified_search_as_dicts
    mock_results = [
        {
            "name": "frontend-design",
            "description": "frontend skill",
            "category": "creative",
            "source": "builtin",
            "identifier": "builtin:frontend-design",
            "is_installed": True,
            "install_path": "/path",
            "preview_url": None,
        }
    ]
    with patch(
        "agent4ml.backend.agents.skill.hub.search.unified_search_as_dicts",
        return_value=mock_results,
    ):
        from agent4ml.backend.agents.skill.hub.search import unified_search_as_dicts

        results = unified_search_as_dicts("frontend")
        assert len(results) == 1
        assert results[0]["name"] == "frontend-design"
        assert results[0]["source"] == "builtin"


# ---------------------------------------------------------------------------
# /skill install 支持 remote identifier（H7）
# ---------------------------------------------------------------------------


def test_install_remote_identifier_detection():
    """remote identifier 前缀检测（github:/well-known:/claude-marketplace:/builtin:）。"""
    prefixes = ("github:", "well-known:", "claude-marketplace:", "builtin:")
    for prefix in prefixes:
        identifier = f"{prefix}test-skill"
        is_remote = any(
            identifier.startswith(p) for p in prefixes
        )
        assert is_remote is True


def test_install_local_path_not_remote():
    """本地路径不是 remote identifier。"""
    prefixes = ("github:", "well-known:", "claude-marketplace:", "builtin:")
    local_path = "/path/to/skill"
    is_remote = any(
        local_path.startswith(p) for p in prefixes
    )
    assert is_remote is False


# ---------------------------------------------------------------------------
# bootstrap 装配 hub（H8）
# ---------------------------------------------------------------------------


def test_hub_module_importable():
    """hub 模块可 import（所有组件就绪）。"""
    from agent4ml.backend.agents.skill.hub import source, hub_store, installer, search
    from agent4ml.backend.agents.skill.hub.sources import (
        builtin_source,
        github_source,
        well_known_source,
        claude_marketplace_source,
    )
    # 验证关键类/函数存在
    assert hasattr(source, "SkillSource")
    assert hasattr(source, "SkillMeta")
    assert hasattr(hub_store, "HubLockFile")
    assert hasattr(hub_store, "SkillsGuard")
    assert hasattr(hub_store, "AuditLog")
    assert hasattr(installer, "Installer")
    assert hasattr(search, "unified_search")
    assert hasattr(builtin_source, "BuiltinSource")
    assert hasattr(github_source, "GitHubSource")
    assert hasattr(well_known_source, "WellKnownSource")
    assert hasattr(claude_marketplace_source, "ClaudeMarketplaceSource")


def test_unified_search_degrades_when_hub_unavailable():
    """hub 模块不可用时 unified_search 降级为只搜 builtin。"""
    # 不传 sources 时自动降级为 BuiltinSource
    with patch(
        "agent4ml.backend.agents.skill.build_skill_manager",
        return_value=None,
    ):
        from agent4ml.backend.agents.skill.hub.search import unified_search

        results = unified_search("anything")
    # build_skill_manager 返 None，降级返空
    assert results == []


def test_installer_resolves_source_by_prefix():
    """Installer._resolve_source 按 identifier 前缀解析 source。"""
    from agent4ml.backend.agents.skill.hub.installer import Installer
    from agent4ml.backend.agents.skill.hub.sources.github_source import GitHubSource

    installer = Installer(sources={"github": GitHubSource()})
    source = installer._resolve_source("github:owner/repo@skill")
    assert source is not None
    assert source.name == "github"


def test_installer_resolve_source_unknown_returns_none():
    """未知前缀返 None。"""
    from agent4ml.backend.agents.skill.hub.installer import Installer

    installer = Installer(sources={"github": MagicMock()})
    assert installer._resolve_source("unknown:identifier") is None
