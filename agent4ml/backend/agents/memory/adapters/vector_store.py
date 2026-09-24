"""VectorStore adapter Protocol — 向量检索加速索引（空壳）。

承接 `Hezao-MemDesign-Docs/agent4ml/00-long-term-memory-foundation.md` §8.2 + §7.7
+ `48-memory-l1-base-layer.md` §4 Step 5.8。

角色：Markdown truth 的 derived shadow index，检索加速用。
非独立 MemoryProvider — 由 HybridRetriever 组合（config.vector_store 启用时注入）。

Layer 1：Protocol 空壳，接口签名完整，可 import 可 mock。
Layer 6：具体实现（ChromaVectorStore / FaissVectorStore / MilvusVectorStore / ...）。

Lifecycle：构造即就绪，shutdown 可选（hasattr duck-type）。
adapter 加载失败 → HybridRetriever no-op 跳过该路召回 + log warning（保证系统不崩）。

INVARIANT:
- adapter 是 Retriever 子组件（非独立 MemoryProvider）
- 叠加模式（与 Markdown truth 共存）
- embedding 由外部计算后填入 trace.embedding（Layer 1 不计算）
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agent4ml.backend.agents.memory.schema import MemoryTrace


@runtime_checkable
class VectorStore(Protocol):
    """VectorStore adapter 协议（Retriever 子组件，叠加模式）。

    角色：Markdown truth 的 derived shadow index，检索加速用。
    非独立 MemoryProvider — 由 HybridRetriever 组合（config.vector_store 启用时注入）。

    Layer 1：Protocol 空壳，接口签名完整，可 import 可 mock。
    Layer 6：具体实现（ChromaVectorStore / FaissVectorStore / ...）。

    Lifecycle：构造即就绪，shutdown 可选（hasattr duck-type）。
    adapter 加载失败 → HybridRetriever no-op 跳过该路召回 + log warning（保证系统不崩）。
    """

    def upsert(self, trace: MemoryTrace) -> None:
        """upsert trace 的 embedding 到 vector index。

        embedding 由外部计算后填入 trace.embedding（Layer 1 不计算）。
        """
        ...

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[tuple[str, float]]:
        """向量检索，返回 (trace_id, similarity) 列表，按 similarity 降序。"""
        ...

    def remove(self, trace_id: str) -> None:
        """删除（遗忘时同步删 index，幂等）。"""
        ...

    def rebuild(self, traces: list[MemoryTrace]) -> None:
        """全量重建（从 Markdown truth 重新索引）。"""
        ...
