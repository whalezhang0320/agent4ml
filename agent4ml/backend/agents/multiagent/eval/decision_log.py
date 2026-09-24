"""L3 DecisionLog — 跨 run specialist 协作 lessons 累积.

设计（43 文档 §4.6 + §11.6 L3-7 + spec.md DecisionLog Requirement）:
- DecisionLogWriter: write_async 异步写（fire-and-forget，不阻塞 L1 turn）
- DecisionLogReader: get_recent_lessons 查最近 N 条 lesson（L2 EvolutionMutator 演化输入样本）
- archive_expired: 超期记录移到 archive 表 + 删除主表（不删除数据，归档保留，L3-7.5 决策 c）
- 不直接注入 prompt（作 EvolutionMutator 输入样本，类似 L2 R2.3 failure cases，L3-7.4 决策 b）
- 依赖 _DecisionLogStore Protocol（Batch 11 L1 MultiAgentMetricsStore 实现此 Protocol）
- DecisionLogRecord 在 eval/types.py（Batch 2 已建）
- ThreadPoolExecutor max_workers=1（fire-and-forget，串行写避免并发冲突）
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from agent4ml.backend.agents.multiagent.eval.types import DecisionLogRecord
from agent4ml.backend.agents.multiagent.evolution.types import FailureCategory


class _DecisionLogStore(Protocol):
    """DecisionLog 持久化 Protocol（Batch 11 L1 MultiAgentMetricsStore 实现）."""

    def save_decision_log(self, record: DecisionLogRecord) -> None: ...

    def get_decision_logs(
        self,
        specialist_name: str,
        failure_category: FailureCategory | None,
        limit: int,
    ) -> list[DecisionLogRecord]: ...

    def archive_decision_logs(self, retention_days: int) -> int: ...


class DecisionLogWriter:
    """异步写 decision log（fire-and-forget，不阻塞 L1 turn）.

    内部调 _DecisionLogStore.save_decision_log（Batch 11 L1 MultiAgentMetricsStore 实现）.
    ThreadPoolExecutor max_workers=1（串行写避免并发冲突）.
    L1 tool handler 调 specialist 后调 write_async（L3-7.1 决策 b）.
    """

    def __init__(self, store: _DecisionLogStore) -> None:
        self._store = store
        self._executor = ThreadPoolExecutor(max_workers=1)

    def write_async(self, record: DecisionLogRecord) -> None:
        """异步写（fire-and-forget，不阻塞调用方）."""
        self._executor.submit(self._store.save_decision_log, record)

    def shutdown(self) -> None:
        """关闭 executor（进程退出时调，等待 pending 写完成）."""
        self._executor.shutdown(wait=True)


class DecisionLogReader:
    """读 decision log（L2 EvolutionMutator 演化时查最近 N 条 lesson）.

    get_recent_lessons: 按 specialist_name + failure_category 过滤 + limit.
    archive_expired: 超期记录移到 archive 表 + 删除主表（L3-7.5 决策 c，不删除数据）.
    """

    def __init__(self, store: _DecisionLogStore) -> None:
        self._store = store

    def get_recent_lessons(
        self,
        specialist_name: str,
        failure_category: FailureCategory,
        limit: int = 5,
    ) -> list[DecisionLogRecord]:
        """L2 EvolutionMutator 演化时查最近 N 条 lesson（作输入样本，不进 prompt）."""
        return self._store.get_decision_logs(specialist_name, failure_category, limit)

    def archive_expired(self, retention_days: int = 90) -> int:
        """超期记录移到 archive 表 + 删除主表（返归档条数）."""
        return self._store.archive_decision_logs(retention_days)
