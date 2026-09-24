"""Retriever Protocol — 检索策略契约。

承接 `Hezao-MemDesign-Docs/agent4ml/00-long-term-memory-foundation.md` §7.5 + §8.2
+ `48-memory-l1-base-layer.md` §4 Step 5.4。

默认实现：HybridRetriever（strategies/default/retriever.py，Layer 3 实现）。
组合（按 config 叠加，非互斥）：
- BM25（总在，从 Markdown truth 索引）
- optional VectorStore（config.vector_store 启用时，derived 检索加速）
- optional GraphStore（config.graph_store 启用时，derived 关联扩散）
四种形态：纯 BM25 / BM25+Vector / BM25+Graph / BM25+Vector+Graph

retrieve 强化（lazy decay + access_count+1）在本实现内完成：
- 检索时按需计算 strength（00 §5.5 Ebbinghaus 公式）
- 检索命中的 trace 自动 access_count+1 + last_accessed 更新（frozen 语义：替换）
- 复合分数：score = similarity × 0.7 + strength × 0.3（00 §5.5）

可替换：VectorRetriever / GraphRetriever / TemporalRetriever（adapters/，Layer 6）。
adapter 加载失败 → no-op 跳过该路召回 + log warning（保证系统不崩）。

INVARIANT: Protocol 纯契约零实现，retrieve 强化在实现里（Layer 3）。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agent4ml.backend.agents.memory.types import MemoryQuery, RetrievalResult


@runtime_checkable
class Retriever(Protocol):
    """检索策略协议（00 §7.5 + §8.2 叠加模式）。"""

    def retrieve(self, query: MemoryQuery) -> list[RetrievalResult]:
        """检索相关记忆，返回按 score 降序排列的结果。

        00 §5.5 复合分数：score = similarity × 0.7 + strength × 0.3
        命中 trace 自动强化（lazy decay + access_count+1）。
        """
        ...
