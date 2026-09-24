"""L2TriggerMiddleware 单测 — 不调 LLM + 不修改 state + 命中 enqueue + 冷却跳过。

设计（spec.md L2TriggerMiddleware Requirement + 42 文档 §7.3）:
- after_model 不修改 ThreadState（返 None，INV-4）
- _should_trigger 纯数值判断（不调 LLM，INV-4）
- 命中阈值 enqueue EvolutionTask
- 1h 冷却未过跳过
- < 1ms 延迟
"""
from __future__ import annotations

import queue
import time
from unittest.mock import MagicMock

import pytest

from agent4ml.backend.agents.multiagent.evolution.metrics_view import (
    GlobalMetricsSnapshot,
    SpecialistMetricsSnapshot,
)
from agent4ml.backend.agents.multiagent.evolution.trigger_manager import (
    TriggerManager,
    TriggerThresholds,
)
from agent4ml.backend.agents.multiagent.evolution.trigger_middleware import L2TriggerMiddleware
from agent4ml.backend.agents.multiagent.evolution.types import (
    FailureCategory,
    TriggerSource,
)


class _MockMetricsView:
    """测试用 MetricsView（可配置返回值）."""

    def __init__(
        self,
        failure_categories: dict[FailureCategory, int] | None = None,
        specialists: dict[str, SpecialistMetricsSnapshot] | None = None,
        global_snap: GlobalMetricsSnapshot | None = None,
    ) -> None:
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


def _make_mw(
    metrics_view: _MockMetricsView,
    cooldown_seconds: float = 3600.0,
) -> L2TriggerMiddleware:
    q: queue.Queue = queue.Queue()
    tm = TriggerManager(task_queue=q, cooldown_seconds=cooldown_seconds)
    return L2TriggerMiddleware(trigger_manager=tm, metrics_view=metrics_view)


# ── L2TriggerMiddleware 不调 LLM ───────────────────────────────────────────────


def test_after_model_not_call_llm():
    """after_model 不调 LLM（INV-4）。查源码无 LLM 调用关键字."""
    from pathlib import Path

    import agent4ml.backend.agents.multiagent.evolution.trigger_middleware as mod

    source = Path(mod.__file__).read_text(encoding="utf-8").lower()
    # 实现文件不应含 LLM 调用关键字
    assert "llm.invoke" not in source
    assert "chat_model" not in source
    assert "bind_tools" not in source


def test_after_model_not_modify_threadstate():
    """after_model 返 None（不修改 ThreadState，INV-4）."""
    mv = _MockMetricsView(
        failure_categories={FailureCategory.CONTEXT_INSUFFICIENT: 10}
    )
    mw = _make_mw(mv)
    state = {"messages": [], "metadata": {"snapshot_id": "snap1"}}
    result = mw.after_model(state, runtime=None)
    assert result is None


def test_after_model_enqueue_on_threshold_hit():
    """命中阈值时 enqueue EvolutionTask."""
    mv = _MockMetricsView(
        failure_categories={FailureCategory.CONTEXT_INSUFFICIENT: 10}
    )
    mw = _make_mw(mv)
    state = {"messages": [], "metadata": {"snapshot_id": "snap1"}}
    mw.after_model(state, runtime=None)
    assert not mw._trigger_manager._queue.empty()
    task = mw._trigger_manager._queue.get_nowait()
    assert task.profile == "default"
    assert task.trigger_source == TriggerSource.FAILURE_FOCUSED


def test_after_model_skip_on_cooldown():
    """1h 内同 profile 触发过 → 不 enqueue（冷却防抖）."""
    mv = _MockMetricsView(
        failure_categories={FailureCategory.CONTEXT_INSUFFICIENT: 10}
    )
    mw = _make_mw(mv, cooldown_seconds=3600.0)
    state = {"messages": [], "metadata": {}}
    # 第一次触发
    mw.after_model(state, runtime=None)
    assert mw._trigger_manager._queue.qsize() == 1
    # 1h 内第二次触发 → 冷却未过跳过
    mw.after_model(state, runtime=None)
    assert mw._trigger_manager._queue.qsize() == 1  # 不增


def test_after_model_no_enqueue_when_no_threshold_hit():
    """无阈值命中 → 不 enqueue."""
    mv = _MockMetricsView(
        failure_categories={FailureCategory.CONTEXT_INSUFFICIENT: 1},  # < 5
        global_snap=GlobalMetricsSnapshot(
            total_calls=1, total_cost_usd=0.1, avg_latency_seconds=10.0,
            total_selections=1, total_completions=1, total_fallbacks=0,
        ),
    )
    mw = _make_mw(mv)
    state = {"messages": [], "metadata": {}}
    mw.after_model(state, runtime=None)
    assert mw._trigger_manager._queue.empty()


def test_extract_snapshot_id_from_metadata():
    mv = _MockMetricsView()
    mw = _make_mw(mv)
    state = {"metadata": {"snapshot_id": "snap123"}}
    assert mw._extract_snapshot_id(state) == "snap123"


def test_extract_snapshot_id_fallback_thread_id():
    mv = _MockMetricsView()
    mw = _make_mw(mv)
    state = {"metadata": {"thread_id": "t1"}}
    assert mw._extract_snapshot_id(state) == "t1"


def test_extract_snapshot_id_empty_when_no_metadata():
    mv = _MockMetricsView()
    mw = _make_mw(mv)
    assert mw._extract_snapshot_id({}) == ""
    assert mw._extract_snapshot_id({"metadata": {}}) == ""
