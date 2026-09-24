"""Installer + unified_search 单测（H5）。

验证：
- Installer: install 流程 + SkillsGuard 拒绝 + uninstall + update
- unified_search: 跨 source 聚合 + 去重 + 排序 + 降级
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent4ml.backend.agents.skill.hub.hub_store import (
    AuditLog,
    HubLockEntry,
    HubLockFile,
    ScanResult,
    SkillsGuard,
)
from agent4ml.backend.agents.skill.hub.installer import Installer
from agent4ml.backend.agents.skill.hub.search import (
    unified_search,
    unified_search_as_dicts,
)
from agent4ml.backend.agents.skill.hub.source import SkillMeta


# ---------------------------------------------------------------------------
# Installer
# ---------------------------------------------------------------------------


def _mock_source(
    name: str = "github",
    fetch_result: Path | None = None,
    search_results: list[SkillMeta] | None = None,
) -> MagicMock:
    """构造 mock SkillSource。"""
    source = MagicMock()
    source.name = name
    source.fetch.return_value = fetch_result or Path("/tmp/skill")
    source.search.return_value = search_results or []
    source.preview.return_value = None
    return source


def _create_mock_skill_dir(tmp_path: Path, content: str = "# Skill\n\nSafe content.") -> Path:
    """创建 mock skill 目录 + SKILL.md。"""
    skill_dir = tmp_path / "fetched-skill"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
    return skill_dir


def test_installer_install_success(tmp_path):
    """install 成功：source.fetch → SkillsGuard.scan → parser.install → HubLockFile.add → AuditLog.append。"""
    skill_dir = _create_mock_skill_dir(tmp_path)
    source = _mock_source(fetch_result=skill_dir)
    lock_file = HubLockFile(lock_path=tmp_path / "lock.json")
    audit_log = AuditLog(log_path=tmp_path / "audit.log")
    guard = SkillsGuard(quarantine_dir=tmp_path / "quarantine")

    installer = Installer(
        sources={"github": source},
        lock_file=lock_file,
        guard=guard,
        audit_log=audit_log,
        dest_root=tmp_path / "skills",
    )

    # mock parser.install
    with patch(
        "agent4ml.backend.agents.skill.parser.install",
        return_value="skill-123",
    ):
        skill_id = installer.install("github:owner/repo@test-skill", name="test-skill")

    assert skill_id == "skill-123"

    # HubLockFile 有 entry
    entry = lock_file.get("test-skill")
    assert entry is not None
    assert entry.source == "github"
    assert entry.identifier == "github:owner/repo@test-skill"

    # AuditLog 有记录
    records = audit_log.read()
    assert len(records) == 1
    assert records[0]["action"] == "install"
    assert records[0]["skill_name"] == "test-skill"


def test_installer_install_skills_guard_rejects(tmp_path):
    """SkillsGuard 拒绝时 install 抛 ValueError。"""
    skill_dir = _create_mock_skill_dir(tmp_path, "# Dangerous\n\nrm -rf /")
    source = _mock_source(fetch_result=skill_dir)
    lock_file = HubLockFile(lock_path=tmp_path / "lock.json")
    audit_log = AuditLog(log_path=tmp_path / "audit.log")
    guard = SkillsGuard(quarantine_dir=tmp_path / "quarantine")

    installer = Installer(
        sources={"github": source},
        lock_file=lock_file,
        guard=guard,
        audit_log=audit_log,
        dest_root=tmp_path / "skills",
    )

    with pytest.raises(ValueError, match="rejected"):
        installer.install("github:unknown/repo@dangerous", name="dangerous")

    # HubLockFile 无 entry
    assert lock_file.get("dangerous") is None
    # AuditLog 无 install 记录
    assert len(audit_log.read()) == 0


def test_installer_install_unknown_source_raises(tmp_path):
    """未知 source 前缀时抛 ValueError。"""
    installer = Installer(sources={"github": _mock_source()})
    with pytest.raises(ValueError, match="Cannot resolve source"):
        installer.install("unknown:identifier", name="test")


def test_installer_uninstall(tmp_path):
    """uninstall 删目录 + HubLockFile.remove + AuditLog.append。"""
    lock_file = HubLockFile(lock_path=tmp_path / "lock.json")
    audit_log = AuditLog(log_path=tmp_path / "audit.log")

    # 先 install 一个
    entry = HubLockEntry(
        name="test-skill",
        source="github",
        identifier="github:owner/repo@test-skill",
        install_path=str(tmp_path / "skills" / "test-skill"),
        installed_at="2026-01-01",
        content_hash="abc",
    )
    lock_file.add(entry)
    # 创建 mock 目录
    skill_path = tmp_path / "skills" / "test-skill"
    skill_path.mkdir(parents=True, exist_ok=True)

    installer = Installer(
        sources={"github": _mock_source()},
        lock_file=lock_file,
        audit_log=audit_log,
        dest_root=tmp_path / "skills",
    )

    installer.uninstall("test-skill")

    # 目录删除
    assert not skill_path.exists()
    # HubLockFile 无 entry
    assert lock_file.get("test-skill") is None
    # AuditLog 有 uninstall 记录
    records = audit_log.read()
    assert len(records) == 1
    assert records[0]["action"] == "uninstall"


def test_installer_uninstall_not_found_raises(tmp_path):
    """uninstall 未注册的 skill 抛 ValueError。"""
    installer = Installer(
        sources={"github": _mock_source()},
        lock_file=HubLockFile(lock_path=tmp_path / "lock.json"),
        audit_log=AuditLog(log_path=tmp_path / "audit.log"),
    )
    with pytest.raises(ValueError, match="not found"):
        installer.uninstall("nonexistent")


def test_installer_update_returns_empty_mvp(tmp_path):
    """MVP：update 返空（需 upstream hash 对比实现）。"""
    installer = Installer(sources={"github": _mock_source()})
    assert installer.update() == []
    assert installer.update("specific-skill") == []


def test_installer_derive_name_from_at():
    """github:owner/repo@skill-name → skill-name。"""
    installer = Installer(sources={})
    assert installer._derive_name("github:owner/repo@my-skill") == "my-skill"


def test_installer_derive_name_from_repo():
    """github:owner/repo → repo。"""
    installer = Installer(sources={})
    assert installer._derive_name("github:owner/my-repo") == "my-repo"


def test_installer_derive_name_from_builtin():
    """builtin:name → name。"""
    installer = Installer(sources={})
    assert installer._derive_name("builtin:tdd") == "tdd"


# ---------------------------------------------------------------------------
# unified_search
# ---------------------------------------------------------------------------


def _mock_meta(name: str, source: str = "builtin", is_installed: bool = False) -> SkillMeta:
    return SkillMeta(
        name=name,
        description=f"desc for {name}",
        category="test",
        source=source,
        identifier=f"{source}:{name}",
        is_installed=is_installed,
    )


def test_unified_search_aggregates_multiple_sources():
    """跨 source 聚合。"""
    source1 = _mock_source(name="builtin", search_results=[_mock_meta("skill-a", "builtin")])
    source2 = _mock_source(name="github", search_results=[_mock_meta("skill-b", "github")])

    results = unified_search("skill", sources=[source1, source2])
    assert len(results) == 2
    names = [r.name for r in results]
    assert "skill-a" in names
    assert "skill-b" in names


def test_unified_search_dedupes_by_name():
    """同名 skill 去重。"""
    source1 = _mock_source(name="builtin", search_results=[_mock_meta("dup", "builtin")])
    source2 = _mock_source(name="github", search_results=[_mock_meta("dup", "github")])

    results = unified_search("dup", sources=[source1, source2])
    assert len(results) == 1  # 去重后只有一个


def test_unified_search_is_installed_priority():
    """is_installed 优先排序。"""
    source = _mock_source(name="builtin", search_results=[
        _mock_meta("not-installed", "builtin", is_installed=False),
        _mock_meta("installed", "builtin", is_installed=True),
    ])
    results = unified_search("skill", sources=[source])
    assert results[0].is_installed is True  # installed 排前面
    assert results[1].is_installed is False


def test_unified_search_limit_truncates():
    """limit 截断。"""
    source = _mock_source(name="builtin", search_results=[
        _mock_meta(f"skill-{i}", "builtin") for i in range(10)
    ])
    results = unified_search("skill", sources=[source], limit=3)
    assert len(results) == 3


def test_unified_search_no_sources_degrades_to_builtin():
    """无 sources 时降级为只搜 builtin。"""
    with patch(
        "agent4ml.backend.agents.skill.build_skill_manager"
    ) as mock_build:
        mock_mgr = MagicMock()
        mock_mgr.search_builtin_skills.return_value = [
            {"name": "tdd", "description": "d", "category": "core", "path": "/p", "is_active": True}
        ]
        mock_build.return_value = mock_mgr
        results = unified_search("tdd")

    assert len(results) == 1
    assert results[0].name == "tdd"
    assert results[0].source == "builtin"


def test_unified_search_source_exception_skipped():
    """source 异常时跳过（不崩）。"""
    bad_source = MagicMock()
    bad_source.name = "bad"
    bad_source.search.side_effect = RuntimeError("boom")
    good_source = _mock_source(name="good", search_results=[_mock_meta("good-skill", "good")])

    results = unified_search("skill", sources=[bad_source, good_source])
    assert len(results) == 1
    assert results[0].name == "good-skill"


def test_unified_search_as_dicts_returns_dict_list():
    """unified_search_as_dicts 返 dict 列表（供 JSON）。"""
    source = _mock_source(name="builtin", search_results=[_mock_meta("tdd", "builtin", is_installed=True)])
    results = unified_search_as_dicts("tdd", sources=[source])
    assert len(results) == 1
    assert isinstance(results[0], dict)
    assert results[0]["name"] == "tdd"
    assert results[0]["source"] == "builtin"
    assert results[0]["is_installed"] is True
    assert "identifier" in results[0]
    assert "install_path" in results[0]
