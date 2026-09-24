"""Skill eval 评估层 types + protocols 单测（L3-E1）。

验证：
- frozen 不可变
- Literal 枚举值正确
- Protocol runtime_checkable
- import 防火墙（eval 不 import app）
- 默认值
"""
from __future__ import annotations

import dataclasses

import pytest

from agent4ml.backend.agents.skill.eval.types import (
    ContractRule,
    EvalRun,
    EvolutionSuggestion,
    SkillHealthReport,
    SkillJudgment,
    TaskQualityScore,
)
from agent4ml.backend.agents.skill.eval.protocols import (
    EvalRunStore,
    ResponseContractChecker,
    RuntimeTracker,
    SkillJudgmentAnalyzer,
    TaskQualityJudge,
)


# ── types: frozen + 默认值 ──────────────────────────────

def test_skill_judgment_frozen():
    j = SkillJudgment(
        judgment_id="j1", skill_id="s1", skill_name="test",
        task_id="t1", skill_applied=True,
    )
    assert j.deviation_note == ""
    assert j.timestamp == ""
    with pytest.raises(dataclasses.FrozenInstanceError):
        j.skill_applied = False  # type: ignore[misc]


def test_evolution_suggestion_defaults():
    s = EvolutionSuggestion(evolution_type="FIX")
    assert s.target_skill_ids == ()
    assert s.direction == ""


def test_evolution_suggestion_captured_no_target():
    s = EvolutionSuggestion(evolution_type="CAPTURED", direction="capture pattern")
    assert s.target_skill_ids == ()
    assert s.direction == "capture pattern"


def test_task_quality_score_frozen():
    s = TaskQualityScore(
        score_id="q1", task_id="t1",
        task_completion=0.9, response_quality=0.8,
        efficiency=0.7, tool_usage=0.8,
        overall_score=0.835,
    )
    assert s.rationale == ""
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.overall_score = 0.5  # type: ignore[misc]


def test_contract_rule_defaults():
    r = ContractRule(rule_id="nonempty", kind="programmatic", hard=True)
    assert r.description == ""
    assert r.params == {}


def test_contract_rule_hard_vs_soft():
    hard = ContractRule(rule_id="nonempty", kind="programmatic", hard=True)
    soft = ContractRule(rule_id="must_cite", kind="programmatic", hard=False)
    assert hard.hard is True
    assert soft.hard is False


def test_contract_rule_llm_binary_kind():
    r = ContractRule(rule_id="no_unfounded_claims", kind="llm_binary", hard=True)
    assert r.kind == "llm_binary"


def test_skill_health_report_defaults():
    r = SkillHealthReport(
        skill_id="s1", skill_name="test",
        window_selections=10,
        applied_rate=0.6, completion_rate=0.5,
        effective_rate=0.4, fallback_rate=0.2,
        trend="degrading",
    )
    assert r.recent_judgments == ()
    assert r.advice == ""


def test_skill_health_report_trend_values():
    for trend in ("improving", "stable", "degrading", "insufficient_data"):
        r = SkillHealthReport(
            skill_id="s1", skill_name="test",
            window_selections=0,
            applied_rate=0.0, completion_rate=0.0,
            effective_rate=0.0, fallback_rate=0.0,
            trend=trend,  # type: ignore[arg-type]
        )
        assert r.trend == trend


def test_eval_run_defaults():
    r = EvalRun(
        eval_run_id="e1", eval_layer="execution",
        skill_ids=("s1",),
    )
    assert r.candidate_id is None
    assert r.baseline_id is None
    assert r.result_json == ""
    assert r.timestamp == ""


def test_eval_run_layer_values():
    for layer in ("execution", "task", "response"):
        r = EvalRun(
            eval_run_id="e1", eval_layer=layer,  # type: ignore[arg-type]
            skill_ids=("s1",),
        )
        assert r.eval_layer == layer


# ── protocols: runtime_checkable ─────────────────────────

def test_skill_judgment_analyzer_protocol():
    assert isinstance(SkillJudgmentAnalyzer, type)


def test_task_quality_judge_protocol():
    assert isinstance(TaskQualityJudge, type)


def test_response_contract_checker_protocol():
    assert isinstance(ResponseContractChecker, type)


def test_runtime_tracker_protocol():
    assert isinstance(RuntimeTracker, type)


def test_eval_run_store_protocol():
    assert isinstance(EvalRunStore, type)


def test_protocol_runtime_checkable_accepts_duck_type():
    """实现 Protocol 方法签名的类应被 isinstance 识别。"""

    class FakeChecker:
        def check(self, candidate_content: str, baseline_content: str):
            ...

    assert isinstance(FakeChecker(), ResponseContractChecker)


def test_protocol_runtime_checkable_rejects_missing_method():
    """缺方法的类不应被识别。"""

    class Incomplete:
        pass

    assert not isinstance(Incomplete(), ResponseContractChecker)


# ── import 防火墙 ────────────────────────────────────────

def test_eval_types_no_app_import():
    """eval 包不 import app（依赖方向：app → agents/skill/eval）。"""
    import agent4ml.backend.agents.skill.eval.types as mod
    source = open(mod.__file__, encoding="utf-8").read()
    assert "from agent4ml.backend.app" not in source
    assert "import app" not in source


def test_eval_protocols_no_app_import():
    import agent4ml.backend.agents.skill.eval.protocols as mod
    source = open(mod.__file__, encoding="utf-8").read()
    assert "from agent4ml.backend.app" not in source
    assert "import app" not in source
