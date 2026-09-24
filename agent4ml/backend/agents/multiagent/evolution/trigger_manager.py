"""TriggerManager — 四源触发 + 1h 冷却 + per-profile 串行（D-1）。

设计（42 文档 §7.5 + spec.md TriggerManager Requirement）:
- 四源：周期（6h 兜底）/ 失败聚焦（24h 窗口某 failure_category ≥ 5）/ specialist 降级（invoked≥5+rate<0.4）/ metric 阈值（cost>$1 或 latency>5min）
- 1h 冷却窗口（per-profile 不重复触发，防抖）
- per-profile 串行锁（daemon thread 单线程消费 queue，不加额外锁，INV-28）
- should_trigger 纯数值判断（不调 LLM），命中返 True + 记 last_trigger_source/detail
- enqueue 写 queue.Queue（daemon thread 消费）
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from agent4ml.backend.agents.multiagent.evolution.metrics_view import MetricsView
from agent4ml.backend.agents.multiagent.evolution.types import (
    EvolutionTask,
    FailureCategory,
    TriggerSource,
)


@dataclass
class TriggerThresholds:
    """触发阈值（R4.4，可配置覆盖默认值）。

    failure_window_seconds: 失败聚焦统计窗口（24h = 86400s）。
    failure_threshold: 失败聚焦触发阈值（≥5 次）。
    degradation_min_invoked: specialist 降级最小调用次数（≥5）。
    degradation_threshold: specialist 降级 completion_rate 阈值（<0.4）。
    cost_alert_usd: 单次 cost 告警（>$1）。
    latency_alert_seconds: 单次 latency 告警（>5min=300s）。
    """

    failure_window_seconds: float = 86400.0  # 24h
    failure_threshold: int = 5
    degradation_min_invoked: int = 5
    degradation_threshold: float = 0.4
    cost_alert_usd: float = 1.0
    latency_alert_seconds: float = 300.0


@dataclass
class TriggerState:
    """per-profile 触发状态（冷却 + last_trigger 记录）。

    last_trigger_ts: 上次触发时间戳（用于 1h 冷却）。
    last_trigger_source: 上次触发的 source（enqueue 时用）。
    last_trigger_detail: 上次触发详情（enqueue 时用）。
    """

    last_trigger_ts: float = 0.0
    last_trigger_source: TriggerSource | None = None
    last_trigger_detail: str = ""


class TriggerManager:
    """四源触发 + 1h 冷却 + per-profile 串行（D-1）。

    should_trigger 纯数值判断（不调 LLM，INV-4）。
    enqueue 写 queue.Queue（daemon thread 消费，per-profile 串行，INV-28）。
    1h 冷却窗口防抖（per-profile 不重复触发，R4.2）。
    """

    def __init__(
        self,
        task_queue: "queue.Queue[EvolutionTask]",
        cooldown_seconds: float = 3600.0,  # 1h
        cron_interval_seconds: float = 21600.0,  # 6h
        thresholds: TriggerThresholds | None = None,
    ) -> None:
        self._queue = task_queue
        self._cooldown_seconds = cooldown_seconds
        self._cron_interval_seconds = cron_interval_seconds
        self._thresholds = thresholds or TriggerThresholds()
        self._lock = threading.Lock()
        self._states: dict[str, TriggerState] = {}
        # last_trigger_source 供 L2TriggerMiddleware._enqueue_evolution_task 读取
        self.last_trigger_source: TriggerSource | None = None
        self.last_trigger_detail: str = ""

    def should_trigger(
        self,
        metrics_view: MetricsView,
        profile: str = "default",
    ) -> bool:
        """Pure numeric check: 4 sources + 1h cooldown (no LLM, INV-4).

        Returns True + records last_trigger_source/detail on hit.
        Returns False on cooldown not expired (no repeat trigger).
        """
        now = time.time()
        with self._lock:
            state = self._states.setdefault(profile, TriggerState())
            # 1h cooldown window (per-profile no repeat trigger)
            if now - state.last_trigger_ts < self._cooldown_seconds:
                return False
            # 4-source check
            source, detail = self._check_sources(metrics_view, now)
            if source is None:
                return False
            state.last_trigger_ts = now
            state.last_trigger_source = source
            state.last_trigger_detail = detail
            self.last_trigger_source = source
            self.last_trigger_detail = detail
            return True

    def _check_sources(
        self,
        metrics_view: MetricsView,
        now: float,
    ) -> tuple[TriggerSource | None, str]:
        """Four-source check (periodic / failure-focused / specialist-degraded / metric-alert).

        Returns (source, detail). source=None means no trigger.
        """
        # 1. 失败聚焦（24h 窗口内某 failure_category ≥ 5）
        since = now - self._thresholds.failure_window_seconds
        cats = metrics_view.get_failure_categories(since=since)
        for cat, count in cats.items():
            if cat in (FailureCategory.GOAL_UNCLEAR, FailureCategory.SANDBOX_ISSUE):
                continue  # 不可演化类不触发
            if count >= self._thresholds.failure_threshold:
                return (
                    TriggerSource.FAILURE_FOCUSED,
                    f"{cat.value}={count} in last 24h",
                )

        # 2. specialist 降级（invoked ≥ 5 + completion_rate < 0.4）
        for name in metrics_view.list_specialists():
            snap = metrics_view.get_specialist_metrics(name, since=since)
            if snap is None:
                continue
            if (
                snap["total_invoked"] >= self._thresholds.degradation_min_invoked
                and snap["completion_rate"] < self._thresholds.degradation_threshold
            ):
                return (
                    TriggerSource.SPECIALIST_DEGRADED,
                    f"{name}: rate={snap['completion_rate']:.2f} invoked={snap['total_invoked']}",
                )

        # 3. metric 阈值（单次 cost > $1 或 latency > 5min）
        global_snap = metrics_view.get_global_metrics(since=since)
        if global_snap["total_cost_usd"] > self._thresholds.cost_alert_usd:
            return (
                TriggerSource.COST_ALERT,
                f"cost=${global_snap['total_cost_usd']:.2f}",
            )
        if global_snap["avg_latency_seconds"] > self._thresholds.latency_alert_seconds:
            return (
                TriggerSource.LATENCY_ALERT,
                f"latency={global_snap['avg_latency_seconds']:.0f}s",
            )

        # 4. 周期兜底（6h cron，由 worker loop time.sleep 兜底，这里不主动触发）
        # 周期触发由 L2EvolutionWorker worker loop 兜底，TriggerManager 不主动判定
        return (None, "")

    def enqueue(self, task: EvolutionTask) -> None:
        """enqueue EvolutionTask 到 queue（daemon thread 消费，per-profile 串行）。

        per-profile 串行由 daemon thread 单线程消费保证（INV-28），不加额外锁。
        """
        self._queue.put(task)
