"""MemoryStore Protocol — 记忆持久化契约。

承接 `Hezao-MemDesign-Docs/agent4ml/00-long-term-memory-foundation.md` §7.5
+ `48-memory-l1-base-layer.md` §4 Step 5.3。

默认实现：MarkdownFileStore（strategies/default/store.py，Layer 3 实现）。
可替换：SQLiteStore / VectorStore / GraphStore（adapters/，Layer 6）。

INVARIANT: Protocol 纯契约零实现，可 mock 可替换（00 D8）。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agent4ml.backend.agents.memory.schema import MemoryTrace, MemoryType
from agent4ml.backend.agents.memory.types import MemoryFilter


@runtime_checkable
class MemoryStore(Protocol):
    """记忆持久化协议（00 §7.5）。

    默认实现 MarkdownFileStore（strategies/default/store.py，Layer 3），
    可替换 SQLiteStore / VectorStore / GraphStore（adapters/，Layer 6）。
    """

    def add(self, trace: MemoryTrace) -> None:
        """新增记忆。trace.id 已存在抛 MemoryConflictError。"""
        ...

    def get(self, trace_id: str) -> MemoryTrace | None:
        """按 id 取记忆，不存在返 None。"""
        ...

    def update(self, trace: MemoryTrace) -> None:
        """更新记忆（frozen 语义：替换）。trace.id 不存在抛 MemoryNotFoundError。"""
        ...

    def batch_update(self, traces: list[MemoryTrace]) -> None:
        """批量更新（frozen 语义：替换）。traces 中任一 id 不存在抛 MemoryNotFoundError。

        用于 consolidate 标记旧 trace forgotten 批量提交（F2 决策：避免 N 次 update 的
        O(N²)，MarkdownFileStore 一次全量重写）。
        """
        ...

    def remove(self, trace_id: str) -> None:
        """删除记忆。不存在静默（幂等）。"""
        ...

    def list_by_type(self, type: MemoryType) -> list[MemoryTrace]:
        """按类型列出。"""
        ...

    def list_by_filter(self, filter: MemoryFilter) -> list[MemoryTrace]:
        """按过滤器列出（遗忘策略用）。"""
        ...

    def list_all(self) -> list[MemoryTrace]:
        """列出全部。"""
        ...
