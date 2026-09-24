"""L3-E8 集成测试：config + IVEFocuser 读 judgment + GitRatchet 接 RuntimeTracker。

验证：
- SkillEvalConfig 从 .env 读取
- IVEFocuser 读 SkillJudgment 历史，deviation_note 进 prompt
- GitRatchet 接受 runtime_tracker 参数
- L3 关闭时 L2 行为不变
"""
from __future__ import annotations

import os

import pytest

from agent4ml.backend.agents.skill.config import SkillConfig, SkillEvalConfig, load_skill_config
from agent4ml.backend.agents.skill.evolution.focus.ive_focuser import IVEFocuser
from agent4ml.backend.agents.skill.evolution.gates.git_ratchet import GitRatchet
from agent4ml.backend.agents.skill.evolution.types import EvolutionContext, FailureEvidence
from agent4ml.backend.agents.skill.eval.runtime_tracker import RuntimeTracker
from agent4ml.backend.agents.skill.eval.types import SkillJudgment
from agent4ml.backend.agents.skill.types import SkillLineage, SkillRecord


# ── SkillEvalConfig ─────────────────────────────────────

def test_eval_config_defaults():
    cfg = SkillEvalConfig()
    assert cfg.enabled is False
    assert cfg.judgment_enabled is True
    assert cfg.task_weights == (0.50, 0.35, 0.05, 0.10)
    assert cfg.runtime_window == 20
    assert cfg.degradation_delta == 0.15


def test_skill_config_has_eval_config():
    cfg = SkillConfig()
    assert isinstance(cfg.eval_config, SkillEvalConfig)
    assert cfg.eval_config.enabled is False


def test_load_skill_config_reads_eval_env(monkeypatch):
    monkeypatch.setenv("AGENT4ML_SKILL_ENABLED", "true")
    monkeypatch.setenv("AGENT4ML_SKILL_EVAL_ENABLED", "true")
    monkeypatch.setenv("AGENT4ML_SKILL_EVAL_RUNTIME_WINDOW", "30")
    monkeypatch.setenv("AGENT4ML_SKILL_EVAL_DEGRADATION_DELTA", "0.2")
    cfg = load_skill_config()
    assert cfg.eval_config.enabled is True
    assert cfg.eval_config.runtime_window == 30
    assert cfg.eval_config.degradation_delta == 0.2


def test_load_skill_config_eval_disabled_by_default(monkeypatch):
    monkeypatch.delenv("AGENT4ML_SKILL_EVAL_ENABLED", raising=False)
    cfg = load_skill_config()
    assert cfg.eval_config.enabled is False


# ── IVEFocuser 读 SkillJudgment ─────────────────────────

class _FakeLLM:
    def __init__(self, content: str):
        self._content = content
        self.last_prompt = ""

    def invoke(self, messages):
        self.last_prompt = messages[0].content
        return type("R", (), {"content": self._content})()


class _FakeStore:
    def __init__(self, judgments=None):
        self._judgments = judgments or []

    def get_judgments(self, skill_id, limit=20):
        return self._judgments[:limit]


def _skill(name="sv"):
    return SkillRecord(
        skill_id=f"{name}__imp", name=name, path="/p", content_hash="h",
        lineage=SkillLineage(origin="IMPORTED"), description="信源验证",
    )


def _fix_ctx():
    return EvolutionContext(
        trigger="METRIC", evolution_type="FIX",
        target_skill=_skill(),
        failure_evidence=(FailureEvidence(
            turn_index=2, tool_name="web_search",
            failure_class="IMPLEMENTATION", description="工具调用失败",
        ),),
    )


def test_ivefocuser_reads_judgment_notes():
    """D-L3-20: IVEFocuser 读 SkillJudgment，deviation_note 进 LLM prompt。"""
    judgments = [
        SkillJudgment(
            judgment_id="j1", skill_id="sv__imp", skill_name="sv",
            task_id="t1", skill_applied=False,
            deviation_note="agent 跳过了 validate quality 步骤",
        ),
    ]
    llm = _FakeLLM('{"class": "IMPLEMENTATION", "direction": "加强 MUST"}')
    focuser = IVEFocuser(llm)
    store = _FakeStore(judgments)

    focuser.focus(_fix_ctx(), store)

    assert "跳过了 validate quality" in llm.last_prompt


def test_ivefocuser_no_judgments_still_works():
    """无 judgment 历史 → 正常工作（不崩）。"""
    llm = _FakeLLM('{"class": "IMPLEMENTATION", "direction": "fix"}')
    focuser = IVEFocuser(llm)
    store = _FakeStore([])

    ctx = focuser.focus(_fix_ctx(), store)
    assert ctx.fix_direction == "fix"


def test_ivefocuser_degrade_includes_judgment_notes():
    """LLM=None 降级时 judgment_notes 进 fix_direction。"""
    judgments = [
        SkillJudgment(
            judgment_id="j1", skill_id="sv__imp", skill_name="sv",
            task_id="t1", skill_applied=False,
            deviation_note="偏差记录1",
        ),
    ]
    focuser = IVEFocuser(None)
    store = _FakeStore(judgments)

    ctx = focuser.focus(_fix_ctx(), store)
    assert "偏差记录1" in ctx.fix_direction


# ── GitRatchet 接 RuntimeTracker ────────────────────────

def test_git_ratchet_accepts_runtime_tracker():
    """D-L3-16: GitRatchet 接受 runtime_tracker 参数。"""
    ratchet = GitRatchet(runtime_tracker=object())
    assert ratchet._runtime_tracker is not None


def test_git_ratchet_without_runtime_tracker():
    ratchet = GitRatchet()
    assert ratchet._runtime_tracker is None


def test_git_ratchet_with_runtime_tracker_does_not_crash():
    """有 runtime_tracker 时不崩（即使 tracker 返空）。"""
    class _FakeStore:
        def get_versions(self, name):
            return []
    class _FakeTracker:
        def degraded_skills(self):
            return []

    ratchet = GitRatchet(runtime_tracker=_FakeTracker())
    skill = SkillRecord(
        skill_id="s1", name="test", path="/p", content_hash="h",
        lineage=SkillLineage(origin="IMPORTED"),
        total_selections=10, total_applied=5,
        total_completions=3, total_fallbacks=2,
    )
    # effective_rate = 3/10 = 0.3 < threshold 0.3 → degraded but no parent to rollback
    result = ratchet.check_and_rollback(_FakeStore(), skill)
    # No versions → None
    assert result is None
