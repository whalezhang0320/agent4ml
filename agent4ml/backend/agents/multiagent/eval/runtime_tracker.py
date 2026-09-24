"""L3 SpecialistRuntimeTracker — specialist 健康监控 + 趋势判定 + degraded 检测.

设计（43 文档 §4.4 + §11.3 L3-4 + spec.md SpecialistRuntimeTracker Requirement）:
- 自建（pattern 复用 skill RuntimeTracker，不实现 skill Protocol——返回类型 SkillHealthReport 字段不兼容）
- health_report(specialist_name, window=20) → SpecialistHealthReport：读 L2 MetricsView 算趋势
- degraded_specialists(threshold=0.4) → list[str]：completion_rate < threshold 且 invoked >= 5
- trigger_l2_evolution_if_degraded(tracker, cron_queue)：degraded 命中 → enqueue L2 cron queue（不直接调 L2 TriggerManager）
- 复用 skill _degradation_delta=0.15 + _MIN_JUDGMENTS_FOR_TREND=4 同构常量
- 复用 L2 MetricsView Protocol（get_specialist_metrics / list_specialists）
- MVP trend 基于 sample_size 判定（前后两半比较需扩展 MetricsView 读历史序列，数据驱动触发后补）
"""
from __future__ import annotations

import queue
from typing import Any

from agent4ml.backend.agents.multiagent.eval.types import SpecialistHealthReport
from agent4ml.backend.agents.multiagent.evolution.metrics_view import MetricsView

_MIN_JUDGMENTS_FOR_TREND = 4
_DEGRADATION_DELTA = 0.15


class SpecialistRuntimeTracker:
    """specialist 健康监控 + 趋势判定 + degraded 检测.

    自建（pattern 复用 skill RuntimeTracker，不实现 skill Protocol——SkillHealthReport 字段不兼容）.
    复用 L2 MetricsView Protocol 读 L1 specialist_records 聚合.
    MVP trend 基于 sample_size 判定（< 4 = insufficient_data / >= 4 = stable）.
    """

    def __init__(
        self,
        metrics_view: MetricsView,
        degradation_delta: float = _DEGRADATION_DELTA,
    ) -> None:
        self._metrics = metrics_view
        self._degradation_delta = degradation_delta

    def health_report(
        self, specialist_name: str, window: int = 20,
    ) -> SpecialistHealthReport | None:
        """读 L2 MetricsView 算趋势 + 构造 SpecialistHealthReport.

        返 None 表示 specialist 不存在或无记录.
        MVP trend: sample_size < _MIN_JUDGMENTS_FOR_TREND → insufficient_data / 否则 stable.
        """
        snapshot = self._metrics.get_specialist_metrics(specialist_name)
        if snapshot is None:
            return None

        if snapshot["sample_size"] < _MIN_JUDGMENTS_FOR_TREND:
            trend = "insufficient_data"
        else:
            trend = "stable"

        fallback_rate = (
            snapshot["total_fallbacks"] / snapshot["total_invoked"]
            if snapshot["total_invoked"] > 0
            else 0.0
        )

        return SpecialistHealthReport(
            specialist_name=specialist_name,
            window_invoked=snapshot["sample_size"],
            completion_rate=snapshot["completion_rate"],
            avg_cost_usd=snapshot["avg_cost_usd"],
            avg_latency_seconds=snapshot["avg_latency_seconds"],
            fallback_rate=fallback_rate,
            trend=trend,  # type: ignore[arg-type]
            advice="",
        )

    def degraded_specialists(self, threshold: float = 0.4) -> list[str]:
        """检测 completion_rate < threshold 且 total_invoked >= 5 的 specialist（L3-4.3 决策 b）."""
        degraded: list[str] = []
        for name in self._metrics.list_specialists():
            snapshot = self._metrics.get_specialist_metrics(name)
            if snapshot is None:
                continue
            if (
                snapshot["total_invoked"] >= 5
                and snapshot["completion_rate"] < threshold
            ):
                degraded.append(name)
        return degraded


def trigger_l2_evolution_if_degraded(
    tracker: SpecialistRuntimeTracker,
    cron_queue: queue.Queue,
    threshold: float = 0.4,
) -> list[str]:
    """degraded 命中 → enqueue L2 cron queue（不直接调 L2 TriggerManager，解耦）.

    返回 degraded specialist name 列表（空列表表示无 degraded）.
    """
    degraded = tracker.degraded_specialists(threshold=threshold)
    for name in degraded:
        cron_queue.put(("specialist_degraded", name))
    return degraded
