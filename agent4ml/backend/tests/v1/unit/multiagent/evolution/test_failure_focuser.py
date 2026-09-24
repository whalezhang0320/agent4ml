"""FailureFocuser 单测 — 按 failure_category 聚类取 top + 过滤不可演化类 + 不调 LLM + dominant_category 判定。

设计（spec.md FailureFocuser Requirement + 42 文档 §7.6 + D-7=c）:
- select_failure_samples：按 failure_category 聚类，每类 top 2，上限 5，过滤不可演化类
- analyze：读 L1 failure_category，分类统计，dominant_category 占比最高可演化类
- 不调 LLM（分类在 L1 已完成，INV-4）
"""
from __future__ import annotations

from pathlib import Path

import pytest

from agent4ml.backend.agents.multiagent.evolution.failure_focuser import (
    FailureFocuser,
    select_failure_samples,
)
from agent4ml.backend.agents.multiagent.evolution.metrics_view import (
    GlobalMetricsSnapshot,
    SpecialistMetricsSnapshot,
)
from agent4ml.backend.agents.multiagent.evolution.types import (
    FailureCategory,
    FailureRecord,
)


class _MockMetricsView:
    """测试用 MetricsView（可配置 failure_categories + recent_failures）。"""

    def __init__(
        self,
        failure_categories=None,
        recent_failures_by_cat=None,
        specialists=None,
        global_snap=None,
    ):
        self._failure_categories = failure_categories or {}
        self._recent_failures = recent_failures_by_cat or {}
        self._specialists = specialists or {}
        self._global_snap = global_snap or GlobalMetricsSnapshot(
            total_calls=0, total_cost_usd=0.0, avg_latency_seconds=0.0,
            total_selections=0, total_completions=0, total_fallbacks=0,
        )

    def get_specialist_metrics(self, name, *, since=None):
        return self._specialists.get(name)

    def get_global_metrics(self, *, since=None):
        return self._global_snap

    def get_failure_categories(self, *, since=None):
        return self._failure_categories

    def get_recent_failures(self, *, category, limit=10):
        return self._recent_failures.get(category, [])[:limit]

    def list_specialists(self):
        return list(self._specialists.keys())


def _make_record(cat: FailureCategory, severity: float = 0.0, name: str = "codex") -> FailureRecord:
    return FailureRecord(
        specialist_name=name, goal="g", success_criteria="sc",
        failure_category=cat, severity=severity,
    )


# ── select_failure_samples ────────────────────────────────────────────────────


def test_select_samples_cluster_by_category():
    """按 failure_category 聚类取 top 2."""
    failures = [
        _make_record(FailureCategory.CONTEXT_INSUFFICIENT, 0.9),
        _make_record(FailureCategory.CONTEXT_INSUFFICIENT, 0.7),
        _make_record(FailureCategory.ABILITY_INSUFFICIENT, 0.5),
    ]
    samples = select_failure_samples(failures, max_per_category=2, max_total=5)
    assert len(samples) == 3
    cats = {s.failure_category for s in samples}
    assert FailureCategory.CONTEXT_INSUFFICIENT in cats
    assert FailureCategory.ABILITY_INSUFFICIENT in cats


def test_select_samples_max_per_category_2():
    """每类上限 2 个."""
    failures = [
        _make_record(FailureCategory.CONTEXT_INSUFFICIENT, 0.9),
        _make_record(FailureCategory.CONTEXT_INSUFFICIENT, 0.7),
        _make_record(FailureCategory.CONTEXT_INSUFFICIENT, 0.5),  # 超过 2 个
    ]
    samples = select_failure_samples(failures, max_per_category=2, max_total=5)
    assert len(samples) == 2  # 只取 top 2
    # severity top 2
    assert samples[0].severity == 0.9
    assert samples[1].severity == 0.7


def test_select_samples_max_total_5():
    """总上限 5 个."""
    failures = []
    for i in range(8):
        failures.append(_make_record(FailureCategory.CONTEXT_INSUFFICIENT, 1.0 - i * 0.1))
    samples = select_failure_samples(failures, max_per_category=2, max_total=5)
    # 每类 top 2 = 2 个，但只有 1 个类，所以 2 个（不超过 5）
    assert len(samples) == 2


def test_select_samples_max_total_5_multiple_categories():
    """多类别总上限 5 个（每类 2 个 × 3 类 = 6，截."""
    failures = []
    for cat in [FailureCategory.CONTEXT_INSUFFICIENT, FailureCategory.ABILITY_INSUFFICIENT]:
        for i in range(3):
            failures.append(_make_record(cat, 1.0 - i * 0.1))
    # 每类 top 2 = 4 个，但加一类会超 5
    # 加第三类
    for i in range(3):
        failures.append(_make_record(FailureCategory.GOAL_UNCLEAR, 1.0 - i * 0.1))
    samples = select_failure_samples(failures, max_per_category=2, max_total=5)
    # GOAL_UNCLEAR 过滤掉，剩 CONTEXT+ABILITY 各 2 = 4 个，不超 5
    assert len(samples) == 4


def test_select_samples_filter_non_evolvable():
    """GOAL_UNCLEAR / SANDBOX_ISSUE 过滤掉（不进入样本）."""
    failures = [
        _make_record(FailureCategory.GOAL_UNCLEAR, 1.0),
        _make_record(FailureCategory.SANDBOX_ISSUE, 1.0),
        _make_record(FailureCategory.CONTEXT_INSUFFICIENT, 0.5),
    ]
    samples = select_failure_samples(failures, max_per_category=2, max_total=5)
    assert len(samples) == 1
    assert samples[0].failure_category == FailureCategory.CONTEXT_INSUFFICIENT


def test_select_samples_empty_input():
    """空输入返空列表."""
    assert select_failure_samples([]) == []


def test_select_samples_all_non_evolvable():
    """全部不可演化类 → 返空列表."""
    failures = [
        _make_record(FailureCategory.GOAL_UNCLEAR, 1.0),
        _make_record(FailureCategory.SANDBOX_ISSUE, 1.0),
    ]
    assert select_failure_samples(failures) == []


def test_select_samples_severity_sort_desc():
    """同类按 severity 降序取 top."""
    failures = [
        _make_record(FailureCategory.CONTEXT_INSUFFICIENT, 0.3),
        _make_record(FailureCategory.CONTEXT_INSUFFICIENT, 0.9),
        _make_record(FailureCategory.CONTEXT_INSUFFICIENT, 0.5),
    ]
    samples = select_failure_samples(failures, max_per_category=2, max_total=5)
    assert len(samples) == 2
    assert samples[0].severity == 0.9  # top 1
    assert samples[1].severity == 0.5  # top 2


# ── FailureFocuser.analyze ─────────────────────────────────────────────────────


def test_analyze_context_dominant():
    """CONTEXT_INSUFFICIENT 8 次 + ABILITY_INSUFFICIENT 3 次 → dominant=CONTEXT."""
    mv = _MockMetricsView(
        failure_categories={
            FailureCategory.CONTEXT_INSUFFICIENT: 8,
            FailureCategory.ABILITY_INSUFFICIENT: 3,
        },
        recent_failures_by_cat={
            FailureCategory.CONTEXT_INSUFFICIENT: [
                _make_record(FailureCategory.CONTEXT_INSUFFICIENT, 0.9),
                _make_record(FailureCategory.CONTEXT_INSUFFICIENT, 0.7),
            ],
            FailureCategory.ABILITY_INSUFFICIENT: [
                _make_record(FailureCategory.ABILITY_INSUFFICIENT, 0.5),
            ],
        },
    )
    focuser = FailureFocuser()
    stats = focuser.analyze(mv)
    assert stats.dominant_category == FailureCategory.CONTEXT_INSUFFICIENT
    assert stats.by_category[FailureCategory.CONTEXT_INSUFFICIENT] == 8
    assert stats.by_category[FailureCategory.ABILITY_INSUFFICIENT] == 3
    # sample_failures 含每类 top 2
    assert FailureCategory.CONTEXT_INSUFFICIENT in stats.sample_failures
    assert FailureCategory.ABILITY_INSUFFICIENT in stats.sample_failures
    assert len(stats.sample_failures[FailureCategory.CONTEXT_INSUFFICIENT]) == 2


def test_analyze_no_dominant_when_all_non_evolvable():
    """GOAL_UNCLEAR / SANDBOX_ISSUE 占主导 → dominant=None（不演化）."""
    mv = _MockMetricsView(
        failure_categories={
            FailureCategory.GOAL_UNCLEAR: 5,
            FailureCategory.SANDBOX_ISSUE: 3,
        },
    )
    focuser = FailureFocuser()
    stats = focuser.analyze(mv)
    assert stats.dominant_category is None


def test_analyze_empty_categories():
    """无失败记录 → dominant=None, sample_failures={}."""
    mv = _MockMetricsView(failure_categories={})
    focuser = FailureFocuser()
    stats = focuser.analyze(mv)
    assert stats.dominant_category is None
    assert stats.sample_failures == {}
    assert stats.by_category == {}


def test_analyze_sample_failures_filtered():
    """sample_failures 不含不可演化类（即使 get_recent_failures 返回）."""
    mv = _MockMetricsView(
        failure_categories={
            FailureCategory.GOAL_UNCLEAR: 5,
            FailureCategory.CONTEXT_INSUFFICIENT: 2,
        },
        recent_failures_by_cat={
            FailureCategory.GOAL_UNCLEAR: [_make_record(FailureCategory.GOAL_UNCLEAR)],
            FailureCategory.CONTEXT_INSUFFICIENT: [_make_record(FailureCategory.CONTEXT_INSUFFICIENT)],
        },
    )
    focuser = FailureFocuser()
    stats = focuser.analyze(mv)
    # sample_failures 不含 GOAL_UNCLEAR
    assert FailureCategory.GOAL_UNCLEAR not in stats.sample_failures
    assert FailureCategory.CONTEXT_INSUFFICIENT in stats.sample_failures


def test_analyze_mixed_dominant_is_max_evolvable():
    """混合时 dominant 是可演化类中占比最高的."""
    mv = _MockMetricsView(
        failure_categories={
            FailureCategory.GOAL_UNCLEAR: 10,  # 不可演化，不计入 dominant
            FailureCategory.ABILITY_INSUFFICIENT: 6,
            FailureCategory.CONTEXT_INSUFFICIENT: 3,
        },
    )
    focuser = FailureFocuser()
    stats = focuser.analyze(mv)
    # ABILITY(6) > CONTEXT(3) → dominant=ABILITY
    assert stats.dominant_category == FailureCategory.ABILITY_INSUFFICIENT


# ── FailureFocuser 不调 LLM ────────────────────────────────────────────────────


def test_failure_focuser_not_call_llm():
    """FailureFocuser 不调 LLM（INV-4）。查源码无 LLM 调用关键字."""
    import agent4ml.backend.agents.multiagent.evolution.failure_focuser as mod

    source = Path(mod.__file__).read_text(encoding="utf-8").lower()
    assert "llm.invoke" not in source
    assert "chat_model" not in source
    assert "bind_tools" not in source
