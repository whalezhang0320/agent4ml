"""SkillManager 门面单测（B9）— load_startup + middleware + build_skill_manager。"""
from __future__ import annotations

import pytest

from agent4ml.backend.agents.skill import SkillManager, build_skill_manager


def _write_skill(tmp_path, name: str) -> None:
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {name} desc\n---\n# {name} body\n",
        encoding="utf-8",
    )


def _env(monkeypatch, tmp_path, *, enabled=True, dirs=None):
    monkeypatch.setenv("AGENT4ML_SKILL_ENABLED", "true" if enabled else "false")
    monkeypatch.setenv("AGENT4ML_SKILL_DB_PATH", str(tmp_path / "skills.db"))
    monkeypatch.setenv("AGENT4ML_SKILL_DIRS", dirs if dirs else str(tmp_path))
    monkeypatch.setenv("AGENT4ML_SKILL_INCLUDE_BUILTIN", "false")  # 隔离 builtin，测用户 skill


def test_load_startup_discovers_and_builds_middleware(tmp_path, monkeypatch):
    _write_skill(tmp_path, "skill-a")
    _env(monkeypatch, tmp_path)
    mgr = build_skill_manager()
    assert mgr is not None
    mgr.load_startup(llm=None)

    assert mgr.get_injection_middleware() is not None
    assert mgr.get_metrics_middleware() is not None
    assert mgr.store is not None
    skills = mgr.list_skills()
    assert len(skills) == 1
    assert skills[0]["name"] == "skill-a"
    assert skills[0]["description"] == "skill-a desc"


def test_load_startup_multiple_skills(tmp_path, monkeypatch):
    _write_skill(tmp_path, "skill-a")
    _write_skill(tmp_path, "skill-b")
    _env(monkeypatch, tmp_path)
    mgr = build_skill_manager()
    mgr.load_startup()
    names = {s["name"] for s in mgr.list_skills()}
    assert names == {"skill-a", "skill-b"}


def test_load_startup_idempotent(tmp_path, monkeypatch):
    _write_skill(tmp_path, "skill-a")
    _env(monkeypatch, tmp_path)
    mgr = build_skill_manager()
    mgr.load_startup()
    mgr.load_startup()  # 二次 discover 不重复注册
    assert len(mgr.list_skills()) == 1


def test_load_startup_includes_builtin_skills(tmp_path, monkeypatch):
    """include_builtin=true 时用户 skill + 核心 builtin skill 均加载。"""
    _write_skill(tmp_path, "skill-a")
    _env(monkeypatch, tmp_path)
    monkeypatch.setenv("AGENT4ML_SKILL_INCLUDE_BUILTIN", "true")  # 覆盖 _env 的 false
    mgr = build_skill_manager()
    mgr.load_startup()
    names = {s["name"] for s in mgr.list_skills()}
    assert "skill-a" in names
    assert "source-verification" in names  # builtin 核心 skill


def test_build_skill_manager_disabled_returns_none(tmp_path, monkeypatch):
    _write_skill(tmp_path, "skill-a")
    _env(monkeypatch, tmp_path, enabled=False)
    assert build_skill_manager() is None


def test_build_skill_manager_no_dir_returns_none(tmp_path, monkeypatch):
    _env(monkeypatch, tmp_path, dirs=str(tmp_path / "nonexistent"))
    assert build_skill_manager() is None


def test_store_persistent_across_load(tmp_path, monkeypatch):
    _write_skill(tmp_path, "skill-a")
    _env(monkeypatch, tmp_path)
    mgr1 = build_skill_manager()
    mgr1.load_startup()
    # 同 DB 重新构造 manager（模拟 switch_expert_mode 不重建 store 语义）
    mgr2 = build_skill_manager()
    mgr2.load_startup()
    assert len(mgr2.list_skills()) == 1


def test_store_property_returns_store(tmp_path, monkeypatch):
    _write_skill(tmp_path, "skill-a")
    _env(monkeypatch, tmp_path)
    mgr = build_skill_manager()
    mgr.load_startup()
    from agent4ml.backend.agents.skill.store import SQLiteSkillStore
    assert isinstance(mgr.store, SQLiteSkillStore)
