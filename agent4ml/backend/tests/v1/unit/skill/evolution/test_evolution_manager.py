"""E8 EvolutionManager 单测 — 闭环编排 + accept/reject/CAPTURED + 手动 evolve/capture。"""
from __future__ import annotations

import pytest

from agent4ml.backend.agents.skill.evolution.manager import EvolutionManager
from agent4ml.backend.agents.skill.evolution.triggers.capture_trigger import CaptureTrigger
from agent4ml.backend.agents.skill.evolution.types import (
    EvalContext,
    EvalResult,
    EvolutionContext,
    GateDecision,
)
from agent4ml.backend.agents.skill.types import SkillLineage, SkillRecord


def _rec(name="sv", skill_id="sv__imp", origin="IMPORTED", parent=()) -> SkillRecord:
    return SkillRecord(
        skill_id=skill_id, name=name, path="/p", content_hash="h",
        lineage=SkillLineage(generation=0, origin=origin, parent_skill_ids=parent),
    )


def _cand(name="sv", skill_id="sv__cand_x", origin="FIXED", parent=("sv__imp",)) -> SkillRecord:
    return SkillRecord(
        skill_id=skill_id, name=name, path="/p2", content_hash="h2",
        is_active=False,
        lineage=SkillLineage(generation=1, origin=origin, parent_skill_ids=parent),
    )


# ── Fakes ───────────────────────────────────────────────


class _FakeStore:
    def __init__(self, active=None):
        self._active = active or {}
        self.create_calls: list = []
        self.evolution_records: list = []

    def list_active(self):
        return list(self._active.values())

    def get_active(self, name):
        return self._active.get(name)

    def get_metrics(self, skill_id):
        return None

    def create_version(self, parent_id, record, origin):
        self.create_calls.append((parent_id, record.skill_id, origin))
        return record.skill_id  # 用 candidate id 作新版本 id

    def record_evolution(self, record):
        self.evolution_records.append(record)
        return record.evolution_id


class _FakeTrigger:
    def __init__(self, contexts):
        self._contexts = contexts
        self.mark_calls: list = []

    def should_trigger(self, store):
        return list(self._contexts)

    def mark_evolved(self, skill_name, total_selections):
        self.mark_calls.append((skill_name, total_selections))


class _FakeFocuser:
    def focus(self, ctx, store):
        return ctx  # 透传


class _FakeMutator:
    def __init__(self, candidate, diff="diff"):
        self._candidate = candidate
        self._diff = diff

    def mutate(self, ctx, llm):
        return self._candidate, self._diff


class _FakeEvalBridge:
    def __init__(self, result):
        self._result = result

    def evaluate(self, ctx):
        return self._result


class _FakeGate:
    def __init__(self, decision):
        self._decision = decision

    def decide(self, candidate, baseline, eval_result):
        return self._decision


class _FakeJournal:
    def __init__(self):
        self.events: list = []

    def append(self, event_type, payload):
        self.events.append((event_type, payload))


def _mgr(store, trigger, gate_decision, candidate=None, eval_score=0.8, journal=None):
    candidate = candidate or _cand()
    return EvolutionManager(
        store=store,
        triggers=[trigger],
        focuser=_FakeFocuser(),
        mutator=_FakeMutator(candidate),
        eval_bridge=_FakeEvalBridge(EvalResult(score=eval_score)),
        gate=_FakeGate(gate_decision),
        llm=None,
        journal=journal,
    )


# ── run_cycle accept ────────────────────────────────────


def test_run_cycle_accept_creates_version_and_records():
    baseline = _rec()
    ctx = EvolutionContext(trigger="METRIC", evolution_type="FIX", target_skill=baseline)
    trigger = _FakeTrigger([ctx])
    store = _FakeStore({"sv": baseline})
    journal = _FakeJournal()
    mgr = _mgr(store, trigger, GateDecision(recommendation="accept", reason="ok", new_version_id="sv__cand_x"), journal=journal)

    records = mgr.run_cycle()
    assert len(records) == 1
    # create_version 调用（parent=baseline, origin=FIXED）
    assert store.create_calls == [("sv__imp", "sv__cand_x", "FIXED")]
    # record_evolution 写入
    assert len(store.evolution_records) == 1
    assert store.evolution_records[0].gate_decision == "accept"
    assert store.evolution_records[0].created_version_id == "sv__cand_x"
    # journal skill.evolve
    assert journal.events[0][0] == "skill.evolve"
    # mark_evolved 调用（anti-loop）
    assert trigger.mark_calls == [("sv", 0)]


def test_run_cycle_reject_no_create_version():
    baseline = _rec()
    ctx = EvolutionContext(trigger="METRIC", evolution_type="FIX", target_skill=baseline)
    trigger = _FakeTrigger([ctx])
    store = _FakeStore({"sv": baseline})
    journal = _FakeJournal()
    mgr = _mgr(store, trigger, GateDecision(recommendation="reject", reason="bad"), journal=journal)

    records = mgr.run_cycle()
    assert len(records) == 1
    assert store.create_calls == []  # reject 不创建版本
    assert store.evolution_records[0].gate_decision == "reject"
    assert store.evolution_records[0].created_version_id is None
    assert journal.events[0][0] == "skill.evolve_rejected"


def test_run_cycle_captured():
    """CAPTURED：无 baseline，create_version parent=""，journal skill.captured。"""
    ctx = EvolutionContext(
        trigger="CAPTURE", evolution_type="CAPTURED", target_skill=None,
        capture_pattern="模式", suggested_name="new-skill",
    )
    trigger = _FakeTrigger([ctx])
    store = _FakeStore()
    journal = _FakeJournal()
    candidate = SkillRecord(
        skill_id="new__cand", name="new-skill", path="/p", content_hash="h",
        is_active=False, lineage=SkillLineage(generation=0, origin="CAPTURED"),
    )
    mgr = _mgr(store, trigger, GateDecision(recommendation="accept", reason="ok", new_version_id="new__cand"), candidate=candidate, journal=journal)

    records = mgr.run_cycle()
    assert len(records) == 1
    assert store.create_calls == [("", "new__cand", "CAPTURED")]  # parent=""
    assert store.evolution_records[0].evolution_type == "CAPTURED"
    assert store.evolution_records[0].baseline_id is None
    assert journal.events[0][0] == "skill.captured"


def test_run_cycle_empty_triggers():
    """无触发 → 空记录。"""
    trigger = _FakeTrigger([])
    store = _FakeStore()
    mgr = _mgr(store, trigger, GateDecision(recommendation="accept", reason="ok"))
    assert mgr.run_cycle() == []


# ── evolve_skill 手动 FIX ───────────────────────────────


def test_evolve_skill_manual_fix():
    baseline = _rec()
    store = _FakeStore({"sv": baseline})
    journal = _FakeJournal()
    mgr = _mgr(store, _FakeTrigger([]), GateDecision(recommendation="accept", reason="ok", new_version_id="sv__cand_x"), journal=journal)

    rec = mgr.evolve_skill("sv")
    assert rec.evolution_type == "FIX"
    assert rec.baseline_id == "sv__imp"
    assert store.create_calls == [("sv__imp", "sv__cand_x", "FIXED")]
    assert journal.events[0][0] == "skill.evolve"


def test_evolve_skill_not_found_raises():
    store = _FakeStore({})
    mgr = _mgr(store, _FakeTrigger([]), GateDecision(recommendation="accept", reason="ok"))
    with pytest.raises(ValueError, match="skill not found"):
        mgr.evolve_skill("nonexistent")


# ── capture_skill 手动 CAPTURED ─────────────────────────


def test_capture_skill_manual():
    store = _FakeStore()
    journal = _FakeJournal()
    candidate = SkillRecord(
        skill_id="new__cand", name="new-skill", path="/p", content_hash="h",
        is_active=False, lineage=SkillLineage(generation=0, origin="CAPTURED"),
    )
    # CaptureTrigger 注册
    capture_trigger = CaptureTrigger()
    mgr = EvolutionManager(
        store=store, triggers=[capture_trigger], focuser=_FakeFocuser(),
        mutator=_FakeMutator(candidate), eval_bridge=_FakeEvalBridge(EvalResult(score=0.7)),
        gate=_FakeGate(GateDecision(recommendation="accept", reason="ok", new_version_id="new__cand")),
        journal=journal,
    )

    rec = mgr.capture_skill("信源交叉核验", "new-skill")
    assert rec.evolution_type == "CAPTURED"
    assert rec.baseline_id is None
    assert store.create_calls == [("", "new__cand", "CAPTURED")]
    assert journal.events[0][0] == "skill.captured"


def test_capture_skill_no_capture_trigger_still_works():
    """无 CaptureTrigger 注册时，capture_skill 自建 context。"""
    store = _FakeStore()
    candidate = SkillRecord(
        skill_id="new__cand", name="new-skill", path="/p", content_hash="h",
        is_active=False, lineage=SkillLineage(generation=0, origin="CAPTURED"),
    )
    mgr = _mgr(store, _FakeTrigger([]), GateDecision(recommendation="accept", reason="ok", new_version_id="new__cand"), candidate=candidate)
    rec = mgr.capture_skill("模式", "new-skill")
    assert rec.evolution_type == "CAPTURED"


# ── journal None 不崩 ───────────────────────────────────


def test_no_journal_no_crash():
    baseline = _rec()
    ctx = EvolutionContext(trigger="METRIC", evolution_type="FIX", target_skill=baseline)
    trigger = _FakeTrigger([ctx])
    store = _FakeStore({"sv": baseline})
    mgr = _mgr(store, trigger, GateDecision(recommendation="accept", reason="ok"), journal=None)
    records = mgr.run_cycle()  # 无 journal 不崩
    assert len(records) == 1
