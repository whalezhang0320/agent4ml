"""B10 集成测试 — AppConfig 字段 / registry slot / middleware 顺序 / 全装配。"""
from __future__ import annotations

import dataclasses

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from agent4ml.backend.agents.capabilities.registry import (
    CapabilityMissingError,
    CapabilityRegistry,
)
from agent4ml.backend.agents.leader.agent import LeaderAgent
from agent4ml.backend.agents.leader.factory import _build_middlewares, make_lead_agent
from agent4ml.backend.agents.middlewares.run_journal_middleware import RunJournalMiddleware
from agent4ml.backend.agents.middlewares.system_context_middleware import (
    SystemContextMiddleware,
)
from agent4ml.backend.agents.middlewares.title_middleware import TitleMiddleware
from agent4ml.backend.agents.reporting.markdown_reporter import MarkdownReporter


class FakeInj:
    pass


class FakeMet:
    pass


def _fake_model() -> FakeListChatModel:
    return FakeListChatModel(responses=["ok"])


def _registry(skill_store=None) -> CapabilityRegistry:
    return CapabilityRegistry(
        models={"researcher": _fake_model()},
        tools={},
        reporter=MarkdownReporter(),
        artifact_store=object(),
        skill_store=skill_store,
    )


# ── AppConfig.skill 字段 ──────────────────────────────────
def test_appconfig_has_skill_field_default_skillconfig():
    from agent4ml.backend.agents.config.schema import AppConfig
    from agent4ml.backend.agents.skill.config import SkillConfig

    fields = {f.name: f for f in dataclasses.fields(AppConfig)}
    assert "skill" in fields
    # 默认 factory 产出 SkillConfig
    cfg = AppConfig(
        name="t", environment="local",
        runtime=dataclasses.fields(AppConfig)[5].default,  # RuntimeConfig default
        models=dataclasses.fields(AppConfig)[6].default,
        tools=dataclasses.fields(AppConfig)[7].default,
        middleware=dataclasses.fields(AppConfig)[8].default,
        reporting=dataclasses.fields(AppConfig)[9].default,
        observability=dataclasses.fields(AppConfig)[10].default,
    )
    assert isinstance(cfg.skill, SkillConfig)


# ── CapabilityRegistry.skill_store slot ──────────────────
def test_registry_skill_store_slot_default_none():
    r = CapabilityRegistry()
    assert r.skill_store is None


def test_registry_get_skill_store_raises_when_none():
    with pytest.raises(CapabilityMissingError, match="skill_store not registered"):
        CapabilityRegistry().get_skill_store()


def test_registry_get_skill_store_returns_when_set():
    sentinel = object()
    r = CapabilityRegistry(skill_store=sentinel)
    assert r.get_skill_store() is sentinel


# ── _build_middlewares skill 中间件挂载顺序 ───────────────
def test_build_middlewares_skill_inserted_after_systemcontext_before_title():
    mws = _build_middlewares(
        expert_mode=False,
        skill_injection_middleware=FakeInj(),
        skill_metrics_middleware=FakeMet(),
    )
    types = [type(m).__name__ for m in mws]
    assert "SystemContextMiddleware" in types
    assert "TitleMiddleware" in types
    assert "FakeInj" in types
    assert "FakeMet" in types
    assert types.index("SystemContextMiddleware") < types.index("FakeInj")
    assert types.index("FakeInj") < types.index("FakeMet")
    assert types.index("FakeMet") < types.index("TitleMiddleware")


def test_build_middlewares_no_skill_when_none():
    mws = _build_middlewares(expert_mode=False)
    types = [type(m).__name__ for m in mws]
    assert "FakeInj" not in types
    assert "FakeMet" not in types


# ── make_lead_agent 透传 skill middleware（用真实 middleware，见 test_full_skill_assembly）──


# ── 全装配：build_skill_manager + registry + make_lead_agent ─
def test_full_skill_assembly(tmp_path, monkeypatch):
    (tmp_path / "skill-a").mkdir()
    (tmp_path / "skill-a" / "SKILL.md").write_text(
        "---\nname: skill-a\ndescription: d\n---\nbody", encoding="utf-8",
    )
    monkeypatch.setenv("AGENT4ML_SKILL_ENABLED", "true")
    monkeypatch.setenv("AGENT4ML_SKILL_DB_PATH", str(tmp_path / "skills.db"))
    monkeypatch.setenv("AGENT4ML_SKILL_DIRS", str(tmp_path))
    monkeypatch.setenv("AGENT4ML_SKILL_INCLUDE_BUILTIN", "false")  # 隔离 builtin

    from agent4ml.backend.agents.skill import build_skill_manager

    mgr = build_skill_manager()
    assert mgr is not None
    mgr.load_startup(llm=None)
    assert len(mgr.list_skills()) == 1

    registry = _registry(skill_store=mgr.store)
    agent = make_lead_agent(
        capability_registry=registry,
        skill_injection_middleware=mgr.get_injection_middleware(),
        skill_metrics_middleware=mgr.get_metrics_middleware(),
    )
    assert isinstance(agent, LeaderAgent)
    assert registry.get_skill_store() is mgr.store


def test_disabled_skill_no_middleware_in_chain(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT4ML_SKILL_ENABLED", "false")
    monkeypatch.setenv("AGENT4ML_SKILL_DB_PATH", str(tmp_path / "skills.db"))
    monkeypatch.setenv("AGENT4ML_SKILL_DIRS", str(tmp_path))
    from agent4ml.backend.agents.skill import build_skill_manager

    assert build_skill_manager() is None
    # make_lead_agent 不传 skill middleware → 链中无
    mws = _build_middlewares(expert_mode=False)
    types = [type(m).__name__ for m in mws]
    assert "SkillInjectionMiddleware" not in types
    assert "SkillMetricsMiddleware" not in types


def test_builtin_skills_start_without_user_skill_directory(tmp_path, monkeypatch):
    """Builtin core skills remain usable when the optional user directory is absent."""
    missing = tmp_path / "does-not-exist"
    monkeypatch.setenv("AGENT4ML_SKILL_ENABLED", "true")
    monkeypatch.setenv("AGENT4ML_SKILL_DB_PATH", str(tmp_path / "skills.db"))
    monkeypatch.setenv("AGENT4ML_SKILL_DIRS", str(missing))
    monkeypatch.setenv("AGENT4ML_SKILL_INCLUDE_BUILTIN", "true")

    from agent4ml.backend.agents.skill import build_skill_manager

    mgr = build_skill_manager()
    assert mgr is not None
    mgr.load_startup(llm=None)
    assert mgr.list_skills()
