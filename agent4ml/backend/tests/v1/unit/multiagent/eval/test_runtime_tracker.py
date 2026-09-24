"""L3 SpecialistRuntimeTracker 单测.

测试要点（结合 L2 MetricsView 联动）:
- health_report 各 trend 分支（insufficient_data / stable）
- health_report 不存在 specialist 返 None
- degraded_specialists 阈值（completion_rate < 0.4 + invoked >= 5）
- invoked < 5 不触发 degraded
- trigger_l2_evolution_if_degraded enqueue 验证
- fallback_rate 计算（0 除保护）
- 复用 L2 MetricsView Protocol（mock 实现）
"""
from __future__ import annotations

import queue
from typing import Any

import pytest

from agent4ml.backend.agents.multiagent.eval.runtime_tracker import (
    SpecialistRuntimeTracker,
    trigger_l2_evolution_if_degraded,
)
from agent4ml.backend.agents.multiagent.eval.types import SpecialistHealthReport
from agent4ml.backend.agents.multiagent.evolution.metrics_view import (
    GlobalMetricsSnapshot,
    SpecialistMetricsSnapshot,
)
from agent4ml.backend.agents.multiagent.evolution.types import FailureCategory


class _MockMetricsView:
    """Mock L2 MetricsView 实现."""

    def __init__(self, specialists: dict[str, SpecialistMetricsSnapshot]) -> None:
        self._specialists = specialists

    def get_specialist_metrics(
        self, name: str, *, since: float | None = None,
    ) -> SpecialistMetricsSnapshot | None:
        return self._specialists.get(name)

    def get_global_metrics(
        self, *, since: float | None = None,
    ) -> GlobalMetricsSnapshot:
        return {
            "total_calls": 0, "total_cost_usd": 0.0,
            "avg_latency_seconds": 0.0, "total_selections": 0,
            "total_completions": 0, "total_fallbacks": 0,
        }

    def get_failure_categories(
        self, *, since: float | None = None,
    ) -> dict[FailureCategory, int]:
        return {}

    def get_recent_failures(
        self, *, category: FailureCategory, limit: int = 10,
    ) -> list[Any]:
        return []

    def list_specialists(self) -> list[str]:
        return list(self._specialists.keys())


def _snapshot(
    name: str, invoked: int = 10, completions: int = 8,
    fallbacks: int = 1, cost: float = 0.5, latency: float = 30.0,
    sample: int | None = None,
) -> SpecialistMetricsSnapshot:
    return {
        "specialist_name": name,
        "total_selections": invoked,
        "total_invoked": invoked,
        "total_completions": completions,
        "total_fallbacks": fallbacks,
        "completion_rate": completions / invoked if invoked > 0 else 0.0,
        "avg_cost_usd": cost,
        "avg_latency_seconds": latency,
        "sample_size": sample if sample is not None else invoked,
    }


class TestHealthReport:
    def test_insufficient_data_trend(self):
        """sample_size < 4 → trend=insufficient_data."""
        mv = _MockMetricsView({"codex": _snapshot("codex", invoked=3, sample=3)})
        tracker = SpecialistRuntimeTracker(mv)
        report = tracker.health_report("codex")
        assert report is not None
        assert report.trend == "insufficient_data"
        assert report.window_invoked == 3

    def test_stable_trend(self):
        """sample_size >= 4 → trend=stable（MVP 简化）."""
        mv = _MockMetricsView({"codex": _snapshot("codex", invoked=10, sample=10)})
        tracker = SpecialistRuntimeTracker(mv)
        report = tracker.health_report("codex")
        assert report is not None
        assert report.trend == "stable"

    def test_nonexistent_specialist_returns_none(self):
        mv = _MockMetricsView({})
        tracker = SpecialistRuntimeTracker(mv)
        assert tracker.health_report("unknown") is None

    def test_fallback_rate_computed(self):
        """fallback_rate 计算（total_fallbacks / total_invoked）."""
        mv = _MockMetricsView({"codex": _snapshot("codex", invoked=10, fallbacks=3)})
        tracker = SpecialistRuntimeTracker(mv)
        report = tracker.health_report("codex")
        assert report is not None
        assert report.fallback_rate == pytest.approx(0.3)

    def test_fallback_rate_zero_division_protection(self):
        """total_invoked=0 时 fallback_rate=0.0（0 除保护）."""
        mv = _MockMetricsView({"codex": _snapshot("codex", invoked=0, fallbacks=0)})
        tracker = SpecialistRuntimeTracker(mv)
        report = tracker.health_report("codex")
        assert report is not None
        assert report.fallback_rate == 0.0


class TestDegradedSpecialists:
    def test_degraded_below_threshold(self):
        """completion_rate < 0.4 且 invoked >= 5 → degraded."""
        mv = _MockMetricsView({
            "codex": _snapshot("codex", invoked=10, completions=2),  # 0.2 < 0.4
            "claude": _snapshot("claude", invoked=10, completions=8),  # 0.8 >= 0.4
        })
        tracker = SpecialistRuntimeTracker(mv)
        degraded = tracker.degraded_specialists(threshold=0.4)
        assert degraded == ["codex"]

    def test_invoked_below_5_not_degraded(self):
        """invoked < 5 不触发 degraded（小样本保护）."""
        mv = _MockMetricsView({
            "codex": _snapshot("codex", invoked=3, completions=0),  # 0.0 < 0.4 但 invoked < 5
        })
        tracker = SpecialistRuntimeTracker(mv)
        assert tracker.degraded_specialists() == []

    def test_no_degraded(self):
        mv = _MockMetricsView({
            "codex": _snapshot("codex", invoked=10, completions=9),
        })
        tracker = SpecialistRuntimeTracker(mv)
        assert tracker.degraded_specialists() == []


class TestTriggerL2Evolution:
    def test_degraded_enqueues(self):
        """degraded 命中 → enqueue L2 cron queue."""
        mv = _MockMetricsView({
            "codex": _snapshot("codex", invoked=10, completions=1),
        })
        tracker = SpecialistRuntimeTracker(mv)
        cron_queue: queue.Queue = queue.Queue()
        degraded = trigger_l2_evolution_if_degraded(tracker, cron_queue)
        assert degraded == ["codex"]
        assert not cron_queue.empty()
        item = cron_queue.get_nowait()
        assert item == ("specialist_degraded", "codex")

    def test_no_degraded_no_enqueue(self):
        mv = _MockMetricsView({
            "codex": _snapshot("codex", invoked=10, completions=9),
        })
        tracker = SpecialistRuntimeTracker(mv)
        cron_queue: queue.Queue = queue.Queue()
        degraded = trigger_l2_evolution_if_degraded(tracker, cron_queue)
        assert degraded == []
        assert cron_queue.empty()
