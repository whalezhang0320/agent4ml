"""E10 集成测试 — bootstrap 装配 + 端到端 CAPTURED 闭环 + disabled 不影响。"""
from __future__ import annotations

from pathlib import Path

import pytest

from agent4ml.backend.agents.skill import SkillManager
from agent4ml.backend.agents.skill.config import SkillConfig
from agent4ml.backend.agents.skill.evolution.manager import EvolutionManager
from agent4ml.backend.agents.skill.evolution.triggers.capture_trigger import CaptureTrigger
from agent4ml.backend.agents.skill.evolution.triggers.metric_monitor import (
    MetricMonitorTrigger,
)


class _MockLLM:
    """端到端 mock LLM：CAPTURED 生成 SKILL.md；FIX 编辑返改进 body。"""

    def invoke(self, messages):
        content = messages[0].content if messages else ""
        if "生成完整 SKILL.md" in content or "可复用模式" in content:
            return type("R", (), {"content": (
                "---\nname: captured-skill\ndescription: 一个捕获的研究过程知识\n"
                "allowed-tools:\n  - web_search\n---\n"
                "## 核心要点\n验证信源可信度。来源：https://example.com。\n\n"
                "## 步骤\n1. 找 2 源\n2. 比对\n\nMUST 交叉核验。NEVER 单源采信。\n"
            )})()
        if "修复方向" in content or "编辑后的 body" in content:
            # FIX 编辑：返改进 body（含 cite/conclusion/directive 提分）
            return type("R", (), {"content": (
                "## 核心要点\n验证信源可信度。来源：https://example.com。\n\n"
                "## 步骤\n1. 找 2 源\n2. 比对\n\nMUST 交叉核验。NEVER 单源采信。\n"
            )})()
        # IVEFocuser / MetricMonitor confirm
        return type("R", (), {"content": "yes"})()


class _FakeJournal:
    def __init__(self):
        self.events = []

    def append(self, event_type, payload):
        self.events.append((event_type, payload))


def _skill_config(tmp_path, evolve=True) -> SkillConfig:
    return SkillConfig(
        enabled=True,
        db_path=str(tmp_path / "skills.db"),
        skill_dirs=(str(tmp_path / "skills"),),
        include_builtin=False,
        evolve_enabled=evolve,
    )


# ── _build_evolution_manager 装配 ───────────────────────


def test_build_evolution_manager_wires_components(tmp_path, monkeypatch):
    """bootstrap helper 正确装配所有组件。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "skills").mkdir()
    mgr = SkillManager(_skill_config(tmp_path))
    mgr.load_startup(llm=None)
    journal = _FakeJournal()

    from agent4ml.backend.app.bootstrap import _build_evolution_manager
    evo = _build_evolution_manager(mgr, _MockLLM(), journal)

    assert isinstance(evo, EvolutionManager)
    assert len(evo._triggers) == 2
    assert any(isinstance(t, MetricMonitorTrigger) for t in evo._triggers)
    assert any(isinstance(t, CaptureTrigger) for t in evo._triggers)
    assert evo._eval_bridge is not None
    assert evo._gate is not None


def test_skill_manager_set_evolution_via_build(tmp_path, monkeypatch):
    """bootstrap 装配后 SkillManager.get_evolution_manager() 非 None。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "skills").mkdir()
    mgr = SkillManager(_skill_config(tmp_path))
    mgr.load_startup(llm=None)
    journal = _FakeJournal()

    from agent4ml.backend.app.bootstrap import _build_evolution_manager
    evo = _build_evolution_manager(mgr, _MockLLM(), journal)
    mgr.set_evolution_manager(evo)

    assert mgr.get_evolution_manager() is evo


# ── 端到端 CAPTURED 闭环 ────────────────────────────────


def test_capture_skill_end_to_end(tmp_path, monkeypatch):
    """CAPTURED 全链路：manual_capture → focus → mutate(LLM生成) → eval → gate → create_version → record。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "skills").mkdir()
    mgr = SkillManager(_skill_config(tmp_path))
    mgr.load_startup(llm=None)
    journal = _FakeJournal()

    from agent4ml.backend.app.bootstrap import _build_evolution_manager
    evo = _build_evolution_manager(mgr, _MockLLM(), journal)
    mgr.set_evolution_manager(evo)

    # 手动 capture
    rec = evo.capture_skill("信源交叉核验", "captured-skill")
    assert rec.evolution_type == "CAPTURED"
    assert rec.skill_name == "captured-skill"
    assert rec.baseline_id is None
    # eval score > 0（programmatic 通过若干 mode）
    assert rec.eval_score > 0
    # gate accept（CAPTURED score>0）
    assert rec.gate_decision == "accept"
    assert rec.created_version_id is not None
    # EvolutionRecord 写入 store
    history = mgr.store.get_evolution_history("captured-skill")
    assert len(history) >= 1
    assert history[0]["evolution_type"] == "CAPTURED"
    # journal skill.captured
    assert any(e[0] == "skill.captured" for e in journal.events)
    # 新 skill 注册到 store（create_version）
    new_rec = mgr.store.get_active("captured-skill")
    assert new_rec is not None
    assert new_rec.lineage.origin == "CAPTURED"


def test_evolve_skill_end_to_end(tmp_path, monkeypatch):
    """FIX 全链路：现有 skill → focus → mutate(LLM编辑) → eval → gate → create_version(FIXED)。"""
    monkeypatch.chdir(tmp_path)
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    # 写一个现有 skill（低质量 body 供 FIX 改进）
    (skills_dir / "my-skill").mkdir()
    (skills_dir / "my-skill" / "SKILL.md").write_text(
        "---\nname: my-skill\ndescription: 测试 skill\n---\n## 步骤\n1. old step\n",
        encoding="utf-8",
    )
    mgr = SkillManager(_skill_config(tmp_path))
    mgr.load_startup(llm=None)
    journal = _FakeJournal()

    from agent4ml.backend.app.bootstrap import _build_evolution_manager
    evo = _build_evolution_manager(mgr, _MockLLM(), journal)
    mgr.set_evolution_manager(evo)

    rec = evo.evolve_skill("my-skill")
    assert rec.evolution_type == "FIX"
    assert rec.baseline_id is not None
    # 新版本注册（FIXED origin）
    versions = mgr.store.get_versions("my-skill")
    assert len(versions) >= 2  # 原 IMPORTED + 新 FIXED
    fixed = [v for v in versions if v.lineage.origin == "FIXED"]
    assert len(fixed) >= 1


# ── disabled 不影响 ─────────────────────────────────────


def test_evolve_disabled_no_evolution_manager(tmp_path, monkeypatch):
    """evolve_enabled=false → SkillManager.get_evolution_manager() None。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "skills").mkdir()
    mgr = SkillManager(_skill_config(tmp_path, evolve=False))
    mgr.load_startup(llm=None)
    assert mgr.get_evolution_manager() is None


def test_run_cycle_no_triggers_empty(tmp_path, monkeypatch):
    """无触发（skill 健康/数据不足）→ run_cycle 返空。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "skills").mkdir()
    mgr = SkillManager(_skill_config(tmp_path))
    mgr.load_startup(llm=None)
    journal = _FakeJournal()

    from agent4ml.backend.app.bootstrap import _build_evolution_manager
    evo = _build_evolution_manager(mgr, _MockLLM(), journal)
    mgr.set_evolution_manager(evo)

    # 无 skill 或 skill 健康 → run_cycle 空
    records = evo.run_cycle()
    assert records == []
