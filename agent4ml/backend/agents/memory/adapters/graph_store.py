"""GraphStore adapter Protocol — 关联扩散激活索引（空壳）。

承接 `Hezao-MemDesign-Docs/agent4ml/00-long-term-memory-foundation.md` §8.2 + §7.7
+ `48-memory-l1-base-layer.md` §4 Step 5.9。

角色：Markdown truth 的 derived shadow index，关联扩散激活用。
非独立 MemoryProvider — 由 HybridRetriever 组合（config.graph_store 启用时注入）。

Layer 1：Protocol 空壳，接口签名完整，可 import 可 mock。
Layer 6：具体实现（Neo4jGraphStore / NetworkXGraphStore / GraphitiGraphStore / ...）。

Lifecycle：构造即就绪，shutdown 可选（hasattr duck-type）。
adapter 加载失败 → HybridRetriever no-op 跳过该路召回 + log warning。

INVARIANT:
- adapter 是 Retriever 子组件（非独立 MemoryProvider）
- 叠加模式（与 Markdown truth 共存）
- 扩散激活（expand）：从 seed trace_ids 出发 BFS，返回关联 trace + 激活分数
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agent4ml.backend.agents.memory.schema import MemoryTrace


@runtime_checkable
class GraphStore(Protocol):
    """GraphStore adapter 协议（Retriever 子组件，叠加模式）。

    角色：Markdown truth 的 derived shadow index，关联扩散激活用。
    非独立 MemoryProvider — 由 HybridRetriever 组合（config.graph_store 启用时注入）。

    Layer 1：Protocol 空壳，接口签名完整，可 import 可 mock。
    Layer 6：具体实现（Neo4jGraphStore / NetworkXGraphStore / ...）。

    Lifecycle：构造即就绪，shutdown 可选（hasattr duck-type）。
    adapter 加载失败 → HybridRetriever no-op 跳过该路召回 + log warning。
    """

    def upsert_node(self, trace: MemoryTrace) -> None:
        """upsert trace 为图节点。"""
        ...

    def upsert_edge(
        self, trace_id_a: str, trace_id_b: str, *,
        strength: float = 0.5, type: str = "related",
    ) -> None:
        """upsert 关联边（Associate 操作同步）。"""
        ...

    def expand(
        self, trace_ids: list[str], *,
        max_depth: int = 2, min_strength: float = 0.3,
    ) -> list[tuple[str, float]]:
        """扩散激活：从 seed trace_ids 出发，返回 (trace_id, activation_score) 列表。"""
        ...

    def remove_node(self, trace_id: str) -> None:
        """删除节点 + 关联边（遗忘时同步，幂等）。"""
        ...

    def rebuild(self, traces: list[MemoryTrace]) -> None:
        """全量重建（从 Markdown truth 重新建图）。"""
        ...
