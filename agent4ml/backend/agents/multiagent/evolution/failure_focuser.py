"""FailureFocuser — 失败聚焦不调 LLM（D-7=c）。

设计（42 文档 §7.6 + spec.md FailureFocuser Requirement）:
- analyze 读 L1 ResultSummarizer 输出的 failure_category 字段（分类在 L1 已完成）
- 分类统计 24h 窗口，按 failure_category 聚类取 top 2 样本（上限 5，INV-17）
- select_failure_samples 过滤 GOAL_UNCLEAR / SANDBOX_ISSUE 不可演化类
- 不调 LLM（INV-4，分类在 L1 ResultSummarizer 已完成）
- dominant_category: 占比最高的可演化类别（不可演化类不作主导）
"""
from __future__ import annotations

from agent4ml.backend.agents.multiagent.evolution.metrics_view import MetricsView
from agent4ml.backend.agents.multiagent.evolution.types import (
    FailureCategory,
    FailureRecord,
    FailureStats,
)

# 不可演化类（不进入 L2 演化流程，转告警）
_NON_EVOLVABLE_CATEGORIES = frozenset({
    FailureCategory.GOAL_UNCLEAR,
    FailureCategory.SANDBOX_ISSUE,
})


def select_failure_samples(
    failures: list[FailureRecord],
    max_per_category: int = 2,
    max_total: int = 5,
) -> list[FailureRecord]:
    """过滤不可演化类 + 按 failure_category 聚类取 top（INV-17）。

    每类取 severity top max_per_category 个，总上限 max_total。
    GOAL_UNCLEAR / SANDBOX_ISSUE 过滤掉（不进入样本）。
    """
    # 过滤不可演化类
    evolvable = [f for f in failures if f.failure_category not in _NON_EVOLVABLE_CATEGORIES]
    # 按 failure_category 聚类
    by_cat: dict[FailureCategory, list[FailureRecord]] = {}
    for f in evolvable:
        by_cat.setdefault(f.failure_category, []).append(f)
    # 每类按 severity 排序取 top max_per_category
    samples: list[FailureRecord] = []
    for cat in sorted(by_cat.keys(), key=lambda c: c.value):
        cat_records = sorted(by_cat[cat], key=lambda r: r.severity, reverse=True)
        samples.extend(cat_records[:max_per_category])
    # 总上限 max_total
    return samples[:max_total]


class FailureFocuser:
    """失败聚焦（D-7=c）。

    analyze 读 L1 ResultSummarizer 输出的 failure_category，分类统计，
    喂给 EvolutionMutator。不调 LLM（分类在 L1 已完成，INV-4）。
    """

    def __init__(self, window_seconds: float = 86400.0) -> None:
        """24h 窗口默认（R4.4a）。"""
        self._window_seconds = window_seconds

    def analyze(
        self,
        metrics_view: MetricsView,
        profile: str = "default",
    ) -> FailureStats:
        """读 L1 failure_category，分类统计 24h 窗口，聚类取 top 样本。

        返 FailureStats（dominant_category + by_category + sample_failures）。
        dominant_category: 占比最高的可演化类别（不可演化类不作主导）。
        """
        import time
        since = time.time() - self._window_seconds
        cats = metrics_view.get_failure_categories(since=since)

        # 按类别取最近失败记录（用于 sample_failures）
        all_failures: list[FailureRecord] = []
        for cat in cats:
            if cat in _NON_EVOLVABLE_CATEGORIES:
                continue  # 不可演化类不取样本
            records = metrics_view.get_recent_failures(category=cat, limit=10)
            all_failures.extend(records)

        # select_failure_samples 聚类取 top
        samples_list = select_failure_samples(all_failures)
        sample_failures: dict[FailureCategory, list[FailureRecord]] = {}
        for s in samples_list:
            sample_failures.setdefault(s.failure_category, []).append(s)

        # dominant_category: 占比最高的可演化类别
        evolvable_cats = {
            cat: count for cat, count in cats.items()
            if cat not in _NON_EVOLVABLE_CATEGORIES
        }
        dominant = (
            max(evolvable_cats, key=evolvable_cats.get) if evolvable_cats else None
        )

        return FailureStats(
            by_category=dict(cats),
            dominant_category=dominant,
            sample_failures=sample_failures,
        )
