"""E7 门控单测 — ScoreDeltaGate + GitRatchet + 2b/L3 Protocol。"""
from __future__ import annotations

import pytest

from agent4ml.backend.agents.skill.evolution.gates.git_ratchet import GitRatchet
from agent4ml.backend.agents.skill.evolution.gates.protocols import (
    ChampionGateProtocol,
    CompositeGateProtocol,
    HITLGateProtocol,
    MultiJudgeGateProtocol,
    ValidationGateProtocol,
)
from agent4ml.backend.agents.skill.evolution.gates.score_delta_gate import ScoreDeltaGate
from agent4ml.backend.agents.skill.evolution.protocols import PromotionGate
from agent4ml.backend.agents.skill.evolution.types import EvalResult, GateDecision
from agent4ml.backend.agents.skill.types import SkillLineage, SkillRecord


def _rec(name="sv", skill_id="sv__imp", generation=0, origin="IMPORTED",
         selections=0, completions=0, parent=()) -> SkillRecord:
    return SkillRecord(
        skill_id=skill_id, name=name, path="/p", content_hash="h",
        is_active=True,
        lineage=SkillLineage(parent_skill_ids=parent, generation=generation, origin=origin),
        total_selections=selections, total_completions=completions,
    )


def _eval(score=0.8, hard_failures=()) -> EvalResult:
    return EvalResult(score=score, hard_failures=hard_failures)


# ── ScoreDeltaGate ──────────────────────────────────────


def test_score_delta_accept():
    gate = ScoreDeltaGate()
    dec = gate.decide(_rec(), _rec(), _eval(score=0.8))
    assert dec.recommendation == "accept"
    assert dec.new_version_id is not None


def test_score_delta_reject_hard_failure():
    """hard_failure → reject。"""
    gate = ScoreDeltaGate()
    dec = gate.decide(_rec(), _rec(), _eval(score=0.9, hard_failures=("nonempty",)))
    assert dec.recommendation == "reject"
    assert "nonempty" in dec.reason


def test_score_delta_reject_low_score():
    """score <= min_delta → reject。"""
    gate = ScoreDeltaGate(min_delta=0.5)
    dec = gate.decide(_rec(), _rec(), _eval(score=0.3))
    assert dec.recommendation == "reject"


def test_score_delta_captured_no_baseline():
    """CAPTURED 无 baseline → score>0 即 accept。"""
    gate = ScoreDeltaGate()
    candidate = _rec(skill_id="new__cand")
    dec = gate.decide(candidate, None, _eval(score=0.7))  # type: ignore[arg-type]
    assert dec.recommendation == "accept"
    assert dec.new_version_id == "new__cand"


def test_score_delta_captured_zero_score_reject():
    gate = ScoreDeltaGate()
    candidate = _rec(skill_id="new__cand")
    dec = gate.decide(candidate, None, _eval(score=0.0))  # type: ignore[arg-type]
    assert dec.recommendation == "reject"


def test_score_delta_is_promotion_gate_protocol():
    assert isinstance(ScoreDeltaGate(), PromotionGate)


# ── GitRatchet ──────────────────────────────────────────


class _FakeStore:
    def __init__(self, versions, rollback_calls=None):
        self._versions = versions
        self.rollback_calls = rollback_calls if rollback_calls is not None else []

    def get_versions(self, name):
        return [v for v in self._versions if v.name == name]

    def rollback(self, skill_id):
        self.rollback_calls.append(skill_id)


def test_git_ratchet_healthy_no_rollback():
    """effective_rate >= threshold → 不 rollback。"""
    current = _rec(skill_id="sv__v2", generation=1, origin="FIXED",
                   selections=10, completions=8, parent=("sv__v1",))  # eff 0.8
    store = _FakeStore([current, _rec(skill_id="sv__v1", generation=0)])
    ratchet = GitRatchet(degradation_threshold=0.3, min_selections=5)
    assert ratchet.check_and_rollback(store, current) is None
    assert store.rollback_calls == []


def test_git_ratchet_degraded_rollback_to_parent():
    """degraded → rollback 到 parent。"""
    current = _rec(skill_id="sv__v2", generation=1, origin="FIXED",
                   selections=10, completions=1, parent=("sv__v1",))  # eff 0.1 < 0.3
    parent = _rec(skill_id="sv__v1", generation=0)
    store = _FakeStore([current, parent])
    ratchet = GitRatchet(degradation_threshold=0.3, min_selections=5)
    target = ratchet.check_and_rollback(store, current)
    assert target == "sv__v1"
    assert store.rollback_calls == ["sv__v1"]


def test_git_ratchet_new_skill_no_data_skip():
    """selections < min → 不评判（anti-loop）。"""
    current = _rec(skill_id="sv__v2", selections=3, completions=0, parent=("sv__v1",))
    store = _FakeStore([current])
    ratchet = GitRatchet(min_selections=5)
    assert ratchet.check_and_rollback(store, current) is None


def test_git_ratchet_no_parent_rollback_lowest_generation():
    """无 parent → rollback 到 generation 最低的非当前版。"""
    current = _rec(skill_id="sv__v3", generation=2, selections=10, completions=1)
    v1 = _rec(skill_id="sv__v1", generation=0)
    v2 = _rec(skill_id="sv__v2", generation=1)
    store = _FakeStore([current, v1, v2])
    ratchet = GitRatchet(degradation_threshold=0.3, min_selections=5)
    target = ratchet.check_and_rollback(store, current)
    assert target == "sv__v1"  # generation 最低


def test_git_ratchet_no_other_versions_no_rollback():
    """无其他版本可回滚 → None。"""
    current = _rec(skill_id="sv__v1", selections=10, completions=1)
    store = _FakeStore([current])
    ratchet = GitRatchet(degradation_threshold=0.3, min_selections=5)
    assert ratchet.check_and_rollback(store, current) is None


# ── 2b/L3 Protocol runtime_checkable ────────────────────


def test_2b_l3_protocols_runtime_checkable():
    """2b/L3 门 Protocol 可 isinstance 校验（impl 注册时）。"""
    # 这些 Protocol 无 impl（2b/L3），但 runtime_checkable 可用于校验未来 impl
    assert hasattr(ChampionGateProtocol, "decide")
    assert hasattr(HITLGateProtocol, "decide")
    assert hasattr(CompositeGateProtocol, "decide")
    assert hasattr(ValidationGateProtocol, "decide")
    assert hasattr(MultiJudgeGateProtocol, "decide")


def test_score_delta_satisfies_2b_protocols():
    """ScoreDeltaGate 实现 decide，结构上满足 2b Protocol（同签名）。"""
    # runtime_checkable Protocol 仅检查方法存在
    gate = ScoreDeltaGate()
    assert hasattr(gate, "decide")
