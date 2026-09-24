"""L3 SpecialistHealthReport + DecisionLogRecord 单测.

测试要点（结合 L2 联动）:
- SpecialistHealthReport frozen 不可变 + 字段默认值 + trend 4 种 Literal
- DecisionLogRecord frozen 不可变 + 字段默认值
- DecisionLogRecord.failure_category 接受 L2 FailureCategory enum（跨模块类型联动）
- DecisionLogRecord.failure_category 接受 None（成功调用无失败分类）
- EvalContext 仍在（Batch 1 不破坏）
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass

import pytest

from agent4ml.backend.agents.multiagent.eval.types import (
    DecisionLogRecord,
    EvalContext,
    SpecialistHealthReport,
    Trend,
)
from agent4ml.backend.agents.multiagent.evolution.promotion_gate import EvalTask
from agent4ml.backend.agents.multiagent.evolution.types import FailureCategory
from types import SimpleNamespace
from typing import Any


def _make_artifact(version: str = "v1") -> Any:
    return SimpleNamespace(
        version=version, template_id="t", artifact_hash=f"h_{version}",
    )


def _make_task(task_id: str = "t1") -> EvalTask:
    return EvalTask(task_id=task_id, goal="g", success_criteria="sc")


# ── SpecialistHealthReport 测试 ──────────────────────────


class TestSpecialistHealthReport:
    def test_frozen_immutable(self):
        report = SpecialistHealthReport(
            specialist_name="codex",
            window_invoked=20,
            completion_rate=0.85,
            avg_cost_usd=0.5,
            avg_latency_seconds=30.0,
            fallback_rate=0.1,
            trend="stable",
        )
        assert is_dataclass(report)
        with pytest.raises(FrozenInstanceError):
            report.advice = "ok"  # type: ignore[misc]

    def test_defaults(self):
        report = SpecialistHealthReport(
            specialist_name="codex",
            window_invoked=10,
            completion_rate=0.8,
            avg_cost_usd=0.3,
            avg_latency_seconds=20.0,
            fallback_rate=0.05,
            trend="improving",
        )
        assert report.advice == ""

    def test_trend_four_values(self):
        """trend 接受 4 种 Literal 值（improving/stable/degrading/insufficient_data）."""
        for trend in ("improving", "stable", "degrading", "insufficient_data"):
            report = SpecialistHealthReport(
                specialist_name="x",
                window_invoked=5,
                completion_rate=0.5,
                avg_cost_usd=0.0,
                avg_latency_seconds=0.0,
                fallback_rate=0.0,
                trend=trend,  # type: ignore[arg-type]
            )
            assert report.trend == trend


# ── DecisionLogRecord 测试 ───────────────────────────────


class TestDecisionLogRecord:
    def test_frozen_immutable(self):
        record = DecisionLogRecord(
            log_id="log1",
            specialist_name="codex",
            task_id="t1",
            goal="g",
            success_criteria="sc",
        )
        assert is_dataclass(record)
        with pytest.raises(FrozenInstanceError):
            record.lesson_text = "lesson"  # type: ignore[misc]

    def test_defaults(self):
        record = DecisionLogRecord(
            log_id="log1",
            specialist_name="codex",
            task_id="t1",
            goal="g",
            success_criteria="sc",
        )
        assert record.failure_category is None
        assert record.success_criteria_met is None
        assert record.lesson_text is None
        assert record.timestamp == ""

    def test_failure_category_accepts_l2_enum(self):
        """failure_category 接受 L2 FailureCategory enum（跨模块类型联动）."""
        record = DecisionLogRecord(
            log_id="log1",
            specialist_name="codex",
            task_id="t1",
            goal="g",
            success_criteria="sc",
            failure_category=FailureCategory.CONTEXT_INSUFFICIENT,
            success_criteria_met=0,
            lesson_text="need more context",
            timestamp="2026-07-28T12:00:00Z",
        )
        assert record.failure_category == FailureCategory.CONTEXT_INSUFFICIENT
        assert record.failure_category.value == "context_insufficient"
        assert record.success_criteria_met == 0

    def test_failure_category_none_for_success(self):
        """failure_category=None 表示成功调用无失败分类."""
        record = DecisionLogRecord(
            log_id="log2",
            specialist_name="claude",
            task_id="t2",
            goal="g2",
            success_criteria="sc2",
            failure_category=None,
            success_criteria_met=1,
        )
        assert record.failure_category is None
        assert record.success_criteria_met == 1


# ── EvalContext 回归（Batch 1 不破坏）──────────────────


class TestEvalContextRegression:
    def test_eval_context_still_works(self):
        """EvalContext（Batch 1 建的）仍正常工作."""
        ctx = EvalContext(
            candidate=_make_artifact("v2"),
            baseline=_make_artifact("v1"),
            task_sample=(_make_task(),),
        )
        assert ctx.candidate.version == "v2"
        assert ctx.baseline.version == "v1"
        assert len(ctx.task_sample) == 1
