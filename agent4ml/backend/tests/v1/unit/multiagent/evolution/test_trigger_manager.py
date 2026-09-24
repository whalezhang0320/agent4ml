"""TriggerManager 单测 — 四源触发 + 1h 冷却 + per-profile 串行 + 阈值边界.

设计（spec.md TriggerManager Requirement + 42 文档 §7.5 + R4.4）:
- 四源：失败聚焦(24h+5)/specialist 降级(invoked≥5+rate<0.4)/cost>$1/latency>5min
- 1h 冷却窗口（per-profile 不重复触发）
- per-profile 串行（daemon thread 单线程消费 queue，不加额外锁）
- should_trigger 纯数值判断（不调 LLM）
"""
from __future__ import annotations

import queue

import pytest

from agent4ml.backend.agents.multiagent.evolution.metrics_view import (
    GlobalMetricsSnapshot,
    SpecialistMetricsSnapshot,
)
from agent4ml.backend.agents.multiagent.evolution.trigger_manager import (
    TriggerManager,
    TriggerThresholds,
)
from agent4ml.backend.agents.multiagent.evolution.types import (
    FailureCategory,
    TriggerSource,
)


class _MockMetricsView:
    def __init__(
        self,
        failure_categories=None,
        specialists=None,
        global_snap=None,
    ):
        self._failure_categories = failure_categories or {}
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
        return []

    def list_specialists(self):
        return list(self._specialists.keys())


def _make_tm(cooldown_seconds=3600.0, thresholds=None):
    q = queue.Queue()
    return TriggerManager(
        task_queue=q, cooldown_seconds=cooldown_seconds, thresholds=thresholds
    )


# ── 四源触发 ───────────────────────────────────────────────────────────────────


def test_failure_focused_trigger_on_threshold():
    """24h 内 CONTEXT_INSUFFICIENT 累计 5 次 → 触发失败聚焦."""
    mv = _MockMetricsView(
        failure_categories={FailureCategory.CONTEXT_INSUFFICIENT: 5}
    )
    tm = _make_tm()
    assert tm.should_trigger(mv) is True
    assert tm.last_trigger_source == TriggerSource.FAILURE_FOCUSED
    assert "context_insufficient=5" in tm.last_trigger_detail


def test_failure_focused_threshold_boundary():
    """阈值边界：4 次不触发，5 次触发."""
    mv = _MockMetricsView(
        failure_categories={FailureCategory.CONTEXT_INSUFFICIENT: 4}
    )
    tm = _make_tm()
    assert tm.should_trigger(mv) is False

    mv2 = _MockMetricsView(
        failure_categories={FailureCategory.CONTEXT_INSUFFICIENT: 5}
    )
    tm2 = _make_tm()
    assert tm2.should_trigger(mv2) is True


def test_failure_focused_skip_non_evolvable_category():
    """GOAL_UNCLEAR / SANDBOX_ISSUE 不触发（不可演化类）."""
    mv = _MockMetricsView(
        failure_categories={FailureCategory.GOAL_UNCLEAR: 10}
    )
    tm = _make_tm()
    assert tm.should_trigger(mv) is False


def test_specialist_degraded_trigger():
    """completion_rate=0.3, invoked=6 → 触发 specialist 降级."""
    mv = _MockMetricsView(
        specialists={
            "codex": SpecialistMetricsSnapshot(
                specialist_name="codex",
                total_selections=10, total_invoked=6, total_completions=2,
                total_fallbacks=4, completion_rate=0.33,
                avg_cost_usd=0.5, avg_latency_seconds=100.0, sample_size=6,
            )
        }
    )
    tm = _make_tm()
    assert tm.should_trigger(mv) is True
    assert tm.last_trigger_source == TriggerSource.SPECIALIST_DEGRADED


def test_specialist_degraded_min_invoked_boundary():
    """invoked < 5 不触发（避免小样本误判）."""
    mv = _MockMetricsView(
        specialists={
            "codex": SpecialistMetricsSnapshot(
                specialist_name="codex",
                total_selections=4, total_invoked=4, total_completions=1,
                total_fallbacks=3, completion_rate=0.25,
                avg_cost_usd=0.5, avg_latency_seconds=100.0, sample_size=4,
            )
        }
    )
    tm = _make_tm()
    assert tm.should_trigger(mv) is False  # invoked=4 < 5


def test_specialist_degraded_rate_boundary():
    """completion_rate=0.4 不触发（< 0.4 才触发）."""
    mv = _MockMetricsView(
        specialists={
            "codex": SpecialistMetricsSnapshot(
                specialist_name="codex",
                total_selections=10, total_invoked=10, total_completions=4,
                total_fallbacks=6, completion_rate=0.4,
                avg_cost_usd=0.5, avg_latency_seconds=100.0, sample_size=10,
            )
        }
    )
    tm = _make_tm()
    assert tm.should_trigger(mv) is False  # rate=0.4 不 < 0.4


def test_cost_alert_trigger():
    """单次 cost > $1 → 触发 cost 告警."""
    mv = _MockMetricsView(
        global_snap=GlobalMetricsSnapshot(
            total_calls=5, total_cost_usd=1.5, avg_latency_seconds=10.0,
            total_selections=5, total_completions=4, total_fallbacks=1,
        )
    )
    tm = _make_tm()
    assert tm.should_trigger(mv) is True
    assert tm.last_trigger_source == TriggerSource.COST_ALERT


def test_latency_alert_trigger():
    """单次 latency > 5min → 触发 latency 告警."""
    mv = _MockMetricsView(
        global_snap=GlobalMetricsSnapshot(
            total_calls=5, total_cost_usd=0.5, avg_latency_seconds=400.0,
            total_selections=5, total_completions=4, total_fallbacks=1,
        )
    )
    tm = _make_tm()
    assert tm.should_trigger(mv) is True
    assert tm.last_trigger_source == TriggerSource.LATENCY_ALERT


def test_no_trigger_when_all_normal():
    """所有源都正常 → 不触发."""
    mv = _MockMetricsView(
        failure_categories={FailureCategory.CONTEXT_INSUFFICIENT: 1},
        specialists={
            "codex": SpecialistMetricsSnapshot(
                specialist_name="codex",
                total_selections=10, total_invoked=10, total_completions=8,
                total_fallbacks=2, completion_rate=0.8,
                avg_cost_usd=0.5, avg_latency_seconds=100.0, sample_size=10,
            )
        },
        global_snap=GlobalMetricsSnapshot(
            total_calls=10, total_cost_usd=0.5, avg_latency_seconds=100.0,
            total_selections=10, total_completions=8, total_fallbacks=2,
        ),
    )
    tm = _make_tm()
    assert tm.should_trigger(mv) is False


# ── 1h 冷却 ───────────────────────────────────────────────────────────────────


def test_cooldown_skip_within_1h():
    """同 profile 1h 内已触发 → 不重复触发（防抖）."""
    mv = _MockMetricsView(
        failure_categories={FailureCategory.CONTEXT_INSUFFICIENT: 10}
    )
    tm = _make_tm(cooldown_seconds=3600.0)
    assert tm.should_trigger(mv) is True
    # 1h 内第二次 → 冷却跳过
    assert tm.should_trigger(mv) is False


def test_cooldown_expired_after_1h():
    """1h 后冷却过期 → 可再次触发."""
    mv = _MockMetricsView(
        failure_categories={FailureCategory.CONTEXT_INSUFFICIENT: 10}
    )
    tm = _make_tm(cooldown_seconds=0.0)  # 0s 冷却，立即过期
    assert tm.should_trigger(mv) is True
    assert tm.should_trigger(mv) is True  # 0s 冷却立即过期


def test_cooldown_per_profile_independent():
    """per-profile 冷却独立（不同 profile 不互相阻塞）."""
    mv = _MockMetricsView(
        failure_categories={FailureCategory.CONTEXT_INSUFFICIENT: 10}
    )
    tm = _make_tm(cooldown_seconds=3600.0)
    assert tm.should_trigger(mv, profile="p1") is True
    # p1 冷却中，p2 不受影响
    assert tm.should_trigger(mv, profile="p2") is True
    # p1 冷却中
    assert tm.should_trigger(mv, profile="p1") is False


# ── per-profile 串行（enqueue）────────────────────────────────────────────────


def test_enqueue_writes_to_queue():
    """enqueue 写 queue.Queue."""
    mv = _MockMetricsView()
    tm = _make_tm()
    from agent4ml.backend.agents.multiagent.evolution.types import EvolutionTask

    task = EvolutionTask(
        task_id="t1", profile="default", trigger_source=TriggerSource.PERIODIC
    )
    tm.enqueue(task)
    assert not tm._queue.empty()
    got = tm._queue.get_nowait()
    assert got.task_id == "t1"


def test_per_profile_serial_by_daemon_thread():
    """per-profile 串行由 daemon thread 单线程消费保证（不加额外锁，INV-28）.

    本测试验证 queue.put 不加锁（多 put 不阻塞），串行由消费端保证.
    """
    mv = _MockMetricsView()
    tm = _make_tm()
    from agent4ml.backend.agents.multiagent.evolution.types import EvolutionTask

    for i in range(5):
        tm.enqueue(EvolutionTask(
            task_id=f"t{i}", profile="default",
            trigger_source=TriggerSource.PERIODIC,
        ))
    assert tm._queue.qsize() == 5  # 5 个任务排队，daemon thread 串行消费


# ── TriggerThresholds 配置覆盖 ───────────────────────────────────────────────


def test_thresholds_custom_config():
    """自定义阈值覆盖默认值."""
    mv = _MockMetricsView(
        failure_categories={FailureCategory.CONTEXT_INSUFFICIENT: 3}
    )
    tm = _make_tm(thresholds=TriggerThresholds(failure_threshold=3))
    assert tm.should_trigger(mv) is True  # 3 次达到自定义阈值


def test_thresholds_defaults():
    t = TriggerThresholds()
    assert t.failure_window_seconds == 86400.0
    assert t.failure_threshold == 5
    assert t.degradation_min_invoked == 5
    assert t.degradation_threshold == 0.4
    assert t.cost_alert_usd == 1.0
    assert t.latency_alert_seconds == 300.0
