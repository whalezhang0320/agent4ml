"""BuiltinSource + GitHubSource 单测（H2）。

验证：
- BuiltinSource: 搜 builtin + is_installed 标记 + fetch/preview
- GitHubSource: identifier 解析 + git clone + GITHUB_TOKEN + cache
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent4ml.backend.agents.skill.hub.sources.builtin_source import BuiltinSource
from agent4ml.backend.agents.skill.hub.sources.github_source import GitHubSource


# ---------------------------------------------------------------------------
# BuiltinSource
# ---------------------------------------------------------------------------


def _mock_skill_manager(results: list[dict]) -> MagicMock:
    mgr = MagicMock()
    mgr.search_builtin_skills.return_value = results
    return mgr


def test_builtin_source_search_returns_skillmeta():
    """BuiltinSource.search 返 SkillMeta 列表。"""
    mgr = _mock_skill_manager([
        {
            "name": "frontend-design",
            "description": "frontend UI design",
            "category": "creative",
            "path": "/builtin/creative/frontend-design/SKILL.md",
            "is_active": True,
        },
    ])
    source = BuiltinSource(skill_manager=mgr)
    results = source.search("frontend")

    assert len(results) == 1
    meta = results[0]
    assert meta.name == "frontend-design"
    assert meta.source == "builtin"
    assert meta.identifier == "builtin:frontend-design"
    assert meta.is_installed is True
    assert meta.install_path == "/builtin/creative/frontend-design/SKILL.md"


def test_builtin_source_search_no_manager_returns_empty():
    """skill_manager 不可用时返空。"""
    with patch(
        "agent4ml.backend.agents.skill.build_skill_manager",
        return_value=None,
    ):
        source = BuiltinSource()
        results = source.search("anything")
    assert results == []


def test_builtin_source_search_limit_truncates():
    """limit 截断。"""
    mgr = _mock_skill_manager([
        {"name": f"skill-{i}", "description": "d", "category": "core", "path": "/p", "is_active": False}
        for i in range(10)
    ])
    source = BuiltinSource(skill_manager=mgr)
    results = source.search("skill", limit=3)
    assert len(results) == 3


def test_builtin_source_fetch_returns_path():
    """builtin fetch 返 skill path（已在本地）。"""
    mgr = _mock_skill_manager([
        {
            "name": "tdd",
            "description": "test driven",
            "category": "core",
            "path": "/builtin/core/tdd/SKILL.md",
            "is_active": True,
        },
    ])
    source = BuiltinSource(skill_manager=mgr)
    path = source.fetch("builtin:tdd", Path("/tmp"))
    # Windows/Linux 路径分隔符差异，用 Path 比较
    assert Path(path) == Path("/builtin/core/tdd/SKILL.md")


def test_builtin_source_name():
    """source.name == "builtin"。"""
    source = BuiltinSource()
    assert source.name == "builtin"


# ---------------------------------------------------------------------------
# GitHubSource
# ---------------------------------------------------------------------------


def test_github_source_parse_identifier_basic():
    """解析 github:owner/repo → (owner, repo, None)。"""
    source = GitHubSource()
    owner, repo, skill_name = source._parse_identifier("github:vercel-labs/agent-skills")
    assert owner == "vercel-labs"
    assert repo == "agent-skills"
    assert skill_name is None


def test_github_source_parse_identifier_with_skill_name():
    """解析 github:owner/repo@skill-name → (owner, repo, skill_name)。"""
    source = GitHubSource()
    owner, repo, skill_name = source._parse_identifier(
        "github:vercel-labs/agent-skills@react-best-practices"
    )
    assert owner == "vercel-labs"
    assert repo == "agent-skills"
    assert skill_name == "react-best-practices"


def test_github_source_parse_identifier_no_prefix():
    """解析 owner/repo（无 github: 前缀）。"""
    source = GitHubSource()
    owner, repo, skill_name = source._parse_identifier("owner/repo")
    assert owner == "owner"
    assert repo == "repo"
    assert skill_name is None


def test_github_source_parse_identifier_invalid():
    """无效 identifier 抛 ValueError。"""
    source = GitHubSource()
    with pytest.raises(ValueError, match="Invalid github identifier"):
        source._parse_identifier("invalid-no-slash")


def test_github_source_build_clone_url_no_token(monkeypatch):
    """无 GITHUB_TOKEN 时用普通 HTTPS URL。"""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    source = GitHubSource()
    url = source._build_clone_url("owner", "repo")
    assert url == "https://github.com/owner/repo.git"


def test_github_source_build_clone_url_with_token(monkeypatch):
    """有 GITHUB_TOKEN 时用 token URL（提升限速）。"""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test_token")
    source = GitHubSource()
    url = source._build_clone_url("owner", "repo")
    assert "ghp_test_token" in url
    assert "github.com/owner/repo.git" in url


def test_github_source_fetch_git_clone(monkeypatch, tmp_path):
    """fetch 调 git clone --depth 1。"""
    source = GitHubSource()
    dest = tmp_path / "test-skill"

    mock_result = MagicMock()
    mock_result.returncode = 0

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        # 创建 mock skill 目录（git clone 后的预期结构）
        dest.mkdir(parents=True, exist_ok=True)
        result_path = source.fetch("github:owner/repo", dest)

    mock_run.assert_called_once()
    call_args = mock_run.call_args
    cmd = call_args[0][0]
    assert "git" in cmd
    assert "clone" in cmd
    assert "--depth" in cmd
    assert "1" in cmd
    assert "https://github.com/owner/repo.git" in cmd


def test_github_source_fetch_with_skill_name(monkeypatch, tmp_path):
    """fetch 含 @skill-name 时返 skill 子目录。"""
    source = GitHubSource()
    dest = tmp_path / "repo"
    skill_dir = dest / "my-skill"

    mock_result = MagicMock()
    mock_result.returncode = 0

    with patch("subprocess.run", return_value=mock_result):
        # 创建 mock skill 子目录
        dest.mkdir(parents=True, exist_ok=True)
        skill_dir.mkdir(parents=True, exist_ok=True)
        result_path = source.fetch("github:owner/repo@my-skill", dest)

    assert result_path == skill_dir


def test_github_source_search_returns_empty_mvp():
    """MVP：GitHubSource.search 返空（需 GitHub API 集成，进阶实现）。"""
    source = GitHubSource()
    results = source.search("anything")
    assert results == []


def test_github_source_preview_returns_none_mvp():
    """MVP：GitHubSource.preview 返 None（需 GitHub Contents API）。"""
    source = GitHubSource()
    assert source.preview("github:owner/repo") is None


def test_github_source_name():
    """source.name == "github"。"""
    source = GitHubSource()
    assert source.name == "github"
