"""E1 evolution/types.py 单测 — frozen 不可变 + 默认值 + CAPTURED 语义。"""
from __future__ import annotations

import dataclasses
import pytest

from agent4ml.backend.agents.skill.evolution.types import (
    EvalContext,
    EvalEvidence,
    EvalResult,
    EvolutionContext,
    EvolutionRecord,
    FailureEvidence,
    GateDecision,
)
from agent4ml.backend.agents.skill.types import SkillLineage, SkillRecord, SkillMetrics


def _record(name: str = "sv", skill_id: str = "sv__imp_a1b2") -> SkillRecord:
    return SkillRecord(
        skill_id=skill_id, name=name, path="/p", content_hash="h",
        lineage=SkillLineage(generation=0, origin="IMPORTED"),
    )


def _metrics() -> SkillMetrics:
    return SkillMetrics(skill_id="sv__imp_a1b2", selections=10, applied=8,
                        completions=6, fallbacks=2,
                        applied_rate=0.8, completion_rate=0.75,
                        effective_rate=0.6, fallback_rate=0.2)


def test_all_frozen():
    """全部 frozen dataclass，修改抛 FrozenInstanceError。"""
    rec = _record()
    fe = FailureEvidence(turn_index=1, tool_name="web_search",
                         failure_class="IMPLEMENTATION", description="d")
    ctx = EvolutionContext(trigger="METRIC", evolution_type="FIX", target_skill=rec)
    ev_ctx = EvalContext(baseline=rec, candidate=rec)
    evi = EvalEvidence(kind="programmatic_rule", rule_name="nonempty",
                       baseline_pass=True, candidate_pass=False)
    res = EvalResult(score=0.8)
    dec = GateDecision(recommendation="accept", reason="r")
    er = EvolutionRecord(evolution_id="e1", skill_name="sv", evolution_type="FIX",
                         trigger="METRIC", baseline_id="sv__imp_a1b2",
                         candidate_id="sv__v1_x", failure_focus="f",
                         mutation_diff="d", eval_score=0.8, gate_decision="accept")

    for obj in (rec, fe, ctx, ev_ctx, evi, res, dec, er):
        with pytest.raises(dataclasses.FrozenInstanceError):
            obj.timestamp = "x"  # type: ignore[misc]


def test_failure_evidence_defaults():
    fe = FailureEvidence(turn_index=None, tool_name=None,
                         failure_class="FUNDAMENTAL", description="d")
    assert fe.impl_fail_count == 0


def test_evolution_context_fix_has_target():
    rec = _record()
    ctx = EvolutionContext(trigger="METRIC", evolution_type="FIX", target_skill=rec)
    assert ctx.target_skill is rec
    assert ctx.failure_evidence == ()
    assert ctx.fix_direction == ""
    assert ctx.capture_pattern == ""
    assert ctx.recent_analyses == ()


def test_evolution_context_captured_no_target():
    """CAPTURED：target_skill=None，capture_pattern + suggested_name 有值。"""
    ctx = EvolutionContext(
        trigger="CAPTURE", evolution_type="CAPTURED", target_skill=None,
        capture_pattern="信源交叉核验", suggested_name="source-cross-check",
    )
    assert ctx.target_skill is None
    assert ctx.capture_pattern == "信源交叉核验"
    assert ctx.suggested_name == "source-cross-check"


def test_eval_context_defaults():
    rec = _record()
    ctx = EvalContext(baseline=rec, candidate=rec)
    assert ctx.metrics_baseline is None
    assert ctx.replay_samples == ()
    assert ctx.task_domain is None


def test_eval_context_with_metrics():
    rec = _record()
    m = _metrics()
    ctx = EvalContext(baseline=rec, candidate=rec, metrics_baseline=m)
    assert ctx.metrics_baseline is m


def test_eval_result_defaults():
    res = EvalResult(score=0.5)
    assert res.metric == "hard"
    assert res.hard_failures == ()
    assert res.evidence == ()
    assert res.confidence == 0.7
    assert res.recommendation == "reject"


def test_eval_result_hard_failures():
    res = EvalResult(score=0.9, hard_failures=("nonempty", "json_parseable"))
    assert res.hard_failures == ("nonempty", "json_parseable")
    assert res.score == 0.9


def test_eval_evidence():
    evi = EvalEvidence(kind="programmatic_rule", rule_name="semantic_density",
                       baseline_pass=True, candidate_pass=False, detail="低密度")
    assert evi.kind == "programmatic_rule"
    assert evi.candidate_pass is False
    assert evi.detail == "低密度"


def test_gate_decision_accept():
    dec = GateDecision(recommendation="accept", reason="score 高", new_version_id="sv__v2")
    assert dec.recommendation == "accept"
    assert dec.new_version_id == "sv__v2"


def test_gate_decision_reject_no_version():
    dec = GateDecision(recommendation="reject", reason="hard_failure")
    assert dec.new_version_id is None


def test_evolution_record_fix():
    er = EvolutionRecord(
        evolution_id="e1", skill_name="sv", evolution_type="FIX", trigger="METRIC",
        baseline_id="sv__imp_a1b2", candidate_id="sv__v1_x",
        failure_focus="指令不清", mutation_diff="- old\n+ new",
        eval_score=0.8, gate_decision="accept", created_version_id="sv__v1_x", timestamp="t",
    )
    assert er.evolution_type == "FIX"
    assert er.baseline_id == "sv__imp_a1b2"
    assert er.created_version_id == "sv__v1_x"


def test_evolution_record_captured_no_baseline():
    """CAPTURED：baseline_id=None（新 skill 无 parent）。"""
    er = EvolutionRecord(
        evolution_id="e2", skill_name="source-cross-check", evolution_type="CAPTURED",
        trigger="CAPTURE", baseline_id=None, candidate_id="scc__v1",
        failure_focus="信源交叉核验模式", mutation_diff="+ full new",
        eval_score=0.75, gate_decision="accept", created_version_id="scc__v1",
    )
    assert er.baseline_id is None
    assert er.evolution_type == "CAPTURED"


def test_evolution_record_reject_no_created():
    er = EvolutionRecord(
        evolution_id="e3", skill_name="sv", evolution_type="FIX", trigger="METRIC",
        baseline_id="sv__imp_a1b2", candidate_id="sv__v1_y",
        failure_focus="f", mutation_diff="d", eval_score=0.2, gate_decision="reject",
    )
    assert er.created_version_id is None
    assert er.timestamp == ""


def test_literal_types_accept_valid():
    """Literal 枚举接受合法值。"""
    ctx = EvolutionContext(trigger="CAPTURE", evolution_type="CAPTURED", target_skill=None)
    assert ctx.trigger == "CAPTURE"
    fe = FailureEvidence(turn_index=0, tool_name="t",
                         failure_class="IMPLEMENTATION", description="d")
    assert fe.failure_class == "IMPLEMENTATION"
    dec = GateDecision(recommendation="pending_human", reason="人审")
    assert dec.recommendation == "pending_human"
