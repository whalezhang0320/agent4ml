"""L2 types 单测 — frozen 不可变 + 字段默认值 + hash 一致性 + Protocol runtime_checkable。

设计（spec.md EvolutionArtifact Requirement + 42 文档 §2.3 + §7.1）:
- EvolutionArtifact runtime_checkable Protocol（isinstance 检查不需显式继承）
- ContextSummaryTemplate / SkillInjectionTemplate frozen 不可变
- artifact_hash 由 payload 计算（相同 payload → 相同 hash，防环用）
- SpecialistCandidate 不含 capability_match（R6.2 修正，LLM 自决）
- FailureCategory 枚举 4 类
- TriggerSource / PromotionDecision 枚举
"""
from __future__ import annotations

import dataclasses

import pytest

from agent4ml.backend.agents.multiagent.evolution.types import (
    BudgetCheckResult,
    BudgetRemaining,
    ContextSummaryTemplate,
    CostRecord,
    EvolutionArtifact,
    EvolutionResult,
    EvolutionTask,
    FailureCategory,
    FailureRecord,
    FailureStats,
    PromotionDecision,
    SkillInjectionTemplate,
    SpecialistCandidate,
    TriggerSource,
)


# ── EvolutionArtifact Protocol runtime_checkable ──────────────────────────────


def test_evolution_artifact_runtime_checkable():
    """ContextSummaryTemplate 实现 EvolutionArtifact Protocol → isinstance 返 True。"""
    t = ContextSummaryTemplate(
        version="v1",
        template_id="default",
        extractors=(),
        filters=(),
        max_tokens=2000,
        prompt_skeleton="skeleton",
    )
    assert isinstance(t, EvolutionArtifact)


def test_skill_injection_template_runtime_checkable():
    """SkillInjectionTemplate 实现 EvolutionArtifact Protocol → isinstance 返 True。"""
    t = SkillInjectionTemplate(
        version="v1",
        template_id="default",
        skill_selector=_DummySelector(),
        injection_format="Skills: {skills}",
    )
    assert isinstance(t, EvolutionArtifact)


def test_evolution_artifact_protocol_not_class_field():
    """EvolutionArtifact 是 Protocol，声明属性接口但不存储数据字段。

    Protocol 的属性是抽象声明（函数定义），非 dataclass field。
    isinstance 检查走 runtime_checkable 结构匹配，非继承。
    """
    import inspect
    # Protocol 类的属性是函数对象（抽象声明），非 dataclass field
    assert not dataclasses.is_dataclass(EvolutionArtifact)
    # Protocol 不应有 dataclass fields（frozen dataclass 实现各自定义字段）
    assert len(dataclasses.fields(EvolutionArtifact)) == 0 if hasattr(EvolutionArtifact, "__dataclass_fields__") else True


# ── ContextSummaryTemplate frozen + 默认值 ──────────────────────────────────


def test_context_summary_template_frozen():
    t = ContextSummaryTemplate(
        version="v1", template_id="default",
        extractors=(), filters=(), max_tokens=2000, prompt_skeleton="s",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.version = "v2"


def test_context_summary_template_fields():
    t = ContextSummaryTemplate(
        version="v1", template_id="default",
        extractors=(), filters=(), max_tokens=2000, prompt_skeleton="s",
    )
    assert t.version == "v1"
    assert t.template_id == "default"
    assert t.extractors == ()
    assert t.filters == ()
    assert t.max_tokens == 2000
    assert t.prompt_skeleton == "s"


def test_context_summary_template_artifact_hash_by_payload():
    """相同 payload → 相同 artifact_hash（防环用，INV-7/INV-27）。"""
    t1 = ContextSummaryTemplate(
        version="v1", template_id="default",
        extractors=(), filters=(), max_tokens=2000, prompt_skeleton="s",
    )
    t2 = ContextSummaryTemplate(
        version="v1", template_id="default",
        extractors=(), filters=(), max_tokens=2000, prompt_skeleton="s",
    )
    assert t1.artifact_hash == t2.artifact_hash


def test_context_summary_template_hash_diff_on_payload_diff():
    """不同 payload → 不同 artifact_hash。"""
    t1 = ContextSummaryTemplate(
        version="v1", template_id="default",
        extractors=(), filters=(), max_tokens=2000, prompt_skeleton="s1",
    )
    t2 = ContextSummaryTemplate(
        version="v1", template_id="default",
        extractors=(), filters=(), max_tokens=2000, prompt_skeleton="s2",
    )
    assert t1.artifact_hash != t2.artifact_hash


def test_context_summary_template_hash_stable_across_runs():
    """hash 计算稳定（同进程多次调用 + 跨进程，sha256 确定性）。"""
    t = ContextSummaryTemplate(
        version="v1", template_id="default",
        extractors=(), filters=(), max_tokens=2000, prompt_skeleton="s",
    )
    h1 = t.artifact_hash
    h2 = t.artifact_hash
    assert h1 == h2
    assert len(h1) == 16  # sha256 hexdigest 前 16 字符


# ── SkillInjectionTemplate frozen + 默认值 + hash ────────────────────────────


def test_skill_injection_template_frozen():
    t = SkillInjectionTemplate(
        version="v1", template_id="default",
        skill_selector=_DummySelector(), injection_format="fmt",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.version = "v2"


def test_skill_injection_template_defaults_max_skills():
    t = SkillInjectionTemplate(
        version="v1", template_id="default",
        skill_selector=_DummySelector(), injection_format="fmt",
    )
    assert t.max_skills == 3


def test_skill_injection_template_hash_by_payload():
    selector = _DummySelector()
    t1 = SkillInjectionTemplate(
        version="v1", template_id="default",
        skill_selector=selector, injection_format="fmt", max_skills=3,
    )
    t2 = SkillInjectionTemplate(
        version="v1", template_id="default",
        skill_selector=selector, injection_format="fmt", max_skills=3,
    )
    assert t1.artifact_hash == t2.artifact_hash


def test_skill_injection_template_hash_diff_on_max_skills():
    selector = _DummySelector()
    t1 = SkillInjectionTemplate(
        version="v1", template_id="default",
        skill_selector=selector, injection_format="fmt", max_skills=3,
    )
    t2 = SkillInjectionTemplate(
        version="v1", template_id="default",
        skill_selector=selector, injection_format="fmt", max_skills=5,
    )
    assert t1.artifact_hash != t2.artifact_hash


# ── FailureCategory 枚举 ───────────────────────────────────────────────────────


def test_failure_category_values():
    assert FailureCategory.GOAL_UNCLEAR.value == "goal_unclear"
    assert FailureCategory.CONTEXT_INSUFFICIENT.value == "context_insufficient"
    assert FailureCategory.ABILITY_INSUFFICIENT.value == "ability_insufficient"
    assert FailureCategory.SANDBOX_ISSUE.value == "sandbox_issue"


def test_failure_category_count():
    assert len(FailureCategory) == 4


# ── FailureRecord / FailureStats ───────────────────────────────────────────────


def test_failure_record_frozen():
    r = FailureRecord(
        specialist_name="codex", goal="g", success_criteria="sc",
        failure_category=FailureCategory.CONTEXT_INSUFFICIENT,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.specialist_name = "x"


def test_failure_record_defaults():
    r = FailureRecord(
        specialist_name="codex", goal="g", success_criteria="sc",
        failure_category=FailureCategory.CONTEXT_INSUFFICIENT,
    )
    assert r.raw_output_tail == ""
    assert r.severity == 0.0
    assert r.timestamp == ""


def test_failure_stats_frozen():
    stats = FailureStats(
        by_category={FailureCategory.CONTEXT_INSUFFICIENT: 1},
        dominant_category=FailureCategory.CONTEXT_INSUFFICIENT,
        sample_failures={},
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        stats.dominant_category = FailureCategory.ABILITY_INSUFFICIENT


def test_failure_stats_dominant_can_be_none():
    """GOAL_UNCLEAR / SANDBOX_ISSUE 占主导时 dominant_category=None（不演化）。"""
    stats = FailureStats(
        by_category={
            FailureCategory.GOAL_UNCLEAR: 5,
            FailureCategory.CONTEXT_INSUFFICIENT: 1,
        },
        dominant_category=None,
        sample_failures={},
    )
    assert stats.dominant_category is None


# ── SpecialistCandidate（不含 capability_match，R6.2 修正）──────────────────


def test_specialist_candidate_frozen():
    c = SpecialistCandidate(
        name="codex", historical_success_rate=0.78,
        avg_cost_usd=0.45, avg_latency_seconds=180, sample_size=20,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.name = "x"


def test_specialist_candidate_no_capability_match_field():
    """SpecialistCandidate 不含 capability_match（R6.2 修正，LLM 自决，INV-35）。"""
    c = SpecialistCandidate(
        name="codex", historical_success_rate=0.78,
        avg_cost_usd=0.45, avg_latency_seconds=180, sample_size=20,
    )
    fields = {f.name for f in dataclasses.fields(c)}
    assert "capability_match" not in fields
    assert "name" in fields
    assert "historical_success_rate" in fields
    assert "avg_cost_usd" in fields
    assert "avg_latency_seconds" in fields
    assert "sample_size" in fields


def test_specialist_candidate_sample_size_low():
    """sample_size < 20 时 LLM 可判断可信度（字段保留，LLM 看 sample_size 自决）。"""
    c = SpecialistCandidate(
        name="codex", historical_success_rate=1.0,
        avg_cost_usd=0.45, avg_latency_seconds=180, sample_size=3,
    )
    assert c.sample_size == 3
    assert c.historical_success_rate == 1.0


# ── CostRecord / BudgetRemaining / BudgetCheckResult ──────────────────────────


def test_cost_record_defaults():
    c = CostRecord()
    assert c.tokens == 0
    assert c.cost_usd == 0.0
    assert c.calls == 1


def test_cost_record_frozen():
    c = CostRecord(tokens=1000, cost_usd=0.5, calls=1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.tokens = 0


def test_budget_remaining_defaults():
    r = BudgetRemaining()
    assert r.tokens == 0
    assert r.cost_usd == 0.0
    assert r.calls == 0


def test_budget_check_result_frozen():
    r = BudgetCheckResult(
        allowed=False, specialist_name="codex",
        reason="daily_cost_exceeded", remaining=None,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.allowed = True


def test_budget_check_result_fallback_target_lead():
    """超限 fallback_target 固定 "lead"（不 fallback 另一 specialist，INV-10）。"""
    r = BudgetCheckResult(
        allowed=False, specialist_name="codex",
        reason="daily_cost_exceeded", remaining=None,
    )
    assert r.fallback_target == "lead"


def test_budget_check_result_allowed_no_reason():
    r = BudgetCheckResult(
        allowed=True, specialist_name="codex", reason=None, remaining=None,
    )
    assert r.allowed is True
    assert r.reason is None


# ── TriggerSource / PromotionDecision 枚举 ────────────────────────────────────


def test_trigger_source_values():
    assert TriggerSource.PERIODIC.value == "periodic"
    assert TriggerSource.FAILURE_FOCUSED.value == "failure_focused"
    assert TriggerSource.SPECIALIST_DEGRADED.value == "specialist_degraded"
    assert TriggerSource.COST_ALERT.value == "cost_alert"
    assert TriggerSource.LATENCY_ALERT.value == "latency_alert"


def test_trigger_source_count():
    assert len(TriggerSource) == 5


def test_promotion_decision_values():
    assert PromotionDecision.ACCEPT.value == "accept"
    assert PromotionDecision.REJECT.value == "reject"
    assert PromotionDecision.FAILED.value == "failed"


# ── EvolutionTask / EvolutionResult ───────────────────────────────────────────


def test_evolution_task_frozen():
    t = EvolutionTask(
        task_id="t1", profile="default",
        trigger_source=TriggerSource.PERIODIC,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.task_id = "x"


def test_evolution_task_defaults():
    t = EvolutionTask(
        task_id="t1", profile="default",
        trigger_source=TriggerSource.PERIODIC,
    )
    assert t.trigger_detail == ""
    assert t.artifact_type == "context_summary"
    assert t.timestamp == ""


def test_evolution_result_frozen():
    r = EvolutionResult(task_id="t1", decision=PromotionDecision.ACCEPT)
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.decision = PromotionDecision.REJECT


def test_evolution_result_defaults():
    r = EvolutionResult(task_id="t1", decision=PromotionDecision.REJECT)
    assert r.new_artifact_id is None
    assert r.rationale == ""
    assert r.error is None


# ── 辅助 ───────────────────────────────────────────────────────────────────────


class _DummySelector:
    """测试用 SkillSelector 实现（Protocol 不需显式继承）。"""

    def select(self, goal: str, available_skills: list) -> tuple:
        return tuple(available_skills[:1])
