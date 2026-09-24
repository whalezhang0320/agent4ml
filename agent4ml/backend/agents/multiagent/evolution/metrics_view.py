"""MetricsView Protocol — L2 读 L1 metrics 接口（D-6.1=A2）。

设计（42 文档 §7.2 + spec.md MetricsView Requirement）:
- runtime_checkable Protocol，L2 只依赖 Protocol，不耦合 OrchestrationStore 实现
- L1 MultiAgentMetricsStore 实现此 Protocol（Batch 12 加方法）
- 5 方法：get_specialist_metrics / get_global_metrics / get_failure_categories / get_recent_failures / list_specialists
- avg_cost_usd / avg_latency_seconds 数据源：JOIN specialist_judgments 表（spec.md multiagent-core Requirement）
"""
from __future__ import annotations

from typing import Any, Protocol, TypedDict, runtime_checkable

from agent4ml.backend.agents.multiagent.evolution.types import FailureCategory, FailureRecord


class SpecialistMetricsSnapshot(TypedDict):
    """单 specialist 聚合 metrics snapshot（get_specialist_metrics 返回）。

    total_*: 4 计数器（selections/invoked/completions/fallbacks）。
    completion_rate: completions / invoked（0 除保护返 0.0）。
    avg_cost_usd / avg_latency_seconds: JOIN specialist_judgments 表算 avg。
    sample_size: 统计样本数（< 20 时 LLM 可判断可信度）。
    """

    specialist_name: str
    total_selections: int
    total_invoked: int
    total_completions: int
    total_fallbacks: int
    completion_rate: float
    avg_cost_usd: float
    avg_latency_seconds: float
    sample_size: int


class GlobalMetricsSnapshot(TypedDict):
    """全局聚合 metrics snapshot（get_global_metrics 返回）。

    total_calls / total_cost_usd / avg_latency_seconds 跨所有 specialist 聚合。
    """

    total_calls: int
    total_cost_usd: float
    avg_latency_seconds: float
    total_selections: int
    total_completions: int
    total_fallbacks: int


@runtime_checkable
class MetricsView(Protocol):
    """L2 读 L1 metrics 接口（Protocol，runtime_checkable）。

    L2 模块只依赖此 Protocol，不 import MultiAgentMetricsStore 实现类。
    L1 MultiAgentMetricsStore 在 Batch 12 实现此 Protocol（加 5 方法）。
    INVARIANT（INV-1）：L2 不直接读 OrchestrationStore，只通过 MetricsView Protocol。
    """

    def get_specialist_metrics(
        self, name: str, *, since: float | None = None
    ) -> SpecialistMetricsSnapshot | None:
        """单 specialist 聚合 metrics snapshot（since=None 查全量，非 None 查 since 后）。

        返 None 表示 specialist 不存在或无记录。
        """
        ...

    def get_global_metrics(
        self, *, since: float | None = None
    ) -> GlobalMetricsSnapshot:
        """全局聚合 metrics snapshot（跨所有 specialist）。"""
        ...

    def get_failure_categories(
        self, *, since: float | None = None
    ) -> dict[FailureCategory, int]:
        """失败分类统计（since 后窗口，按 failure_category 计数）。

        FailureFocuser 读此方法判定 dominant_category。
        """
        ...

    def get_recent_failures(
        self, *, category: FailureCategory, limit: int = 10
    ) -> list[FailureRecord]:
        """最近 N 条某类失败记录（FailureFocuser 取 top 样本用）。"""
        ...

    def list_specialists(self) -> list[str]:
        """所有有记录的 specialist name 列表（candidate metadata 生成用）。"""
        ...
