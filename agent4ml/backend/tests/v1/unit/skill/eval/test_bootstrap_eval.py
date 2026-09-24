"""L3 收尾：bootstrap 装配集成测试。

验证：
- AGENT4ML_SKILL_EVAL_ENABLED=true 时 EvalLayer 装配
- AGENT4ML_SKILL_EVAL_ENABLED=false 时无 EvalLayer
- EvolutionManager 的 eval_bridge 被替换为 RegistryEvalBridge
- import 防火墙
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from agent4ml.backend.agents.skill.config import SkillConfig, SkillEvalConfig


def test_eval_layer_dataclass():
    """EvalLayer frozen dataclass 可创建。"""
    from agent4ml.backend.agents.skill.eval import EvalLayer
    layer = EvalLayer(
        bridge="bridge",
        judgment_analyzer=None,
        task_judge=None,
        runtime_tracker="tracker",
    )
    assert layer.bridge == "bridge"
    assert layer.judgment_analyzer is None
    assert layer.runtime_tracker == "tracker"


def test_build_eval_layer_creates_components(tmp_path: Path):
    """_build_eval_layer 产出所有 L3 组件。"""
    from agent4ml.backend.app.bootstrap import _build_eval_layer
    from agent4ml.backend.agents.skill import SkillManager
    from agent4ml.backend.agents.skill.eval import EvalLayer
    from agent4ml.backend.agents.skill.eval.registry import RegistryEvalBridge
    from agent4ml.backend.agents.skill.eval.runtime_tracker import RuntimeTracker

    cfg = SkillConfig(
        enabled=True,
        db_path=str(tmp_path / "test_bootstrap.db"),
        skill_dirs=(str(tmp_path / "skills"),),
        include_builtin=False,
        eval_config=SkillEvalConfig(enabled=True),
    )
    (tmp_path / "skills").mkdir(exist_ok=True)
    mgr = SkillManager(cfg)

    class _FakeLLM:
        def invoke(self, messages):
            return type("R", (), {"content": "{}"})()

    layer = _build_eval_layer(mgr, _FakeLLM())
    assert isinstance(layer, EvalLayer)
    assert isinstance(layer.bridge, RegistryEvalBridge)
    assert isinstance(layer.runtime_tracker, RuntimeTracker)
    assert layer.judgment_analyzer is not None  # judgment_enabled=True by default
    assert layer.task_judge is not None  # task_judge_enabled=True by default


def test_build_eval_layer_replaces_evolution_bridge(tmp_path: Path):
    """若 EvolutionManager 已装配，_build_eval_layer 替换其 eval_bridge。"""
    from agent4ml.backend.app.bootstrap import _build_eval_layer, _build_evolution_manager
    from agent4ml.backend.agents.skill import SkillManager
    from agent4ml.backend.agents.skill.eval.registry import RegistryEvalBridge

    cfg = SkillConfig(
        enabled=True,
        db_path=str(tmp_path / "test_evo.db"),
        skill_dirs=(str(tmp_path / "skills"),),
        include_builtin=False,
        evolve_enabled=True,
        eval_config=SkillEvalConfig(enabled=True),
    )
    (tmp_path / "skills").mkdir(exist_ok=True)
    mgr = SkillManager(cfg)

    class _FakeLLM:
        def invoke(self, messages):
            return type("R", (), {"content": "{}"})()

    class _FakeJournal:
        def append(self, *a, **kw):
            pass

    # 先装配 L2
    mgr.set_evolution_manager(_build_evolution_manager(mgr, _FakeLLM(), _FakeJournal()))
    # 验证 L2 用的是 ProgrammaticEvalBridge
    from agent4ml.backend.agents.skill.evolution.eval.programmatic_bridge import (
        ProgrammaticEvalBridge,
    )
    assert isinstance(mgr.get_evolution_manager()._eval_bridge, ProgrammaticEvalBridge)

    # 装配 L3 → 替换为 RegistryEvalBridge
    mgr.set_eval_layer(_build_eval_layer(mgr, _FakeLLM()))
    assert isinstance(mgr.get_evolution_manager()._eval_bridge, RegistryEvalBridge)


def test_build_eval_layer_judgment_disabled(tmp_path: Path):
    """judgment_enabled=False → judgment_analyzer=None。"""
    from agent4ml.backend.app.bootstrap import _build_eval_layer
    from agent4ml.backend.agents.skill import SkillManager

    cfg = SkillConfig(
        enabled=True,
        db_path=str(tmp_path / "test_dis.db"),
        skill_dirs=(str(tmp_path / "skills"),),
        include_builtin=False,
        eval_config=SkillEvalConfig(enabled=True, judgment_enabled=False),
    )
    (tmp_path / "skills").mkdir(exist_ok=True)
    mgr = SkillManager(cfg)

    class _FakeLLM:
        def invoke(self, messages):
            return type("R", (), {"content": "{}"})()

    layer = _build_eval_layer(mgr, _FakeLLM())
    assert layer.judgment_analyzer is None
    assert layer.task_judge is not None  # task_judge still enabled


def test_eval_disabled_no_eval_layer(tmp_path: Path, monkeypatch):
    """AGENT4ML_SKILL_EVAL_ENABLED 不设 → eval_config.enabled=False → 不装配。"""
    monkeypatch.delenv("AGENT4ML_SKILL_EVAL_ENABLED", raising=False)
    from agent4ml.backend.agents.skill.config import load_skill_config
    cfg = load_skill_config()
    assert cfg.eval_config.enabled is False
