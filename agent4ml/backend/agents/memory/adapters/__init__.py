"""记忆 adapter 包（VectorStore / GraphStore Protocol 空壳）。

叠加模式：与 Markdown truth 共存，作为 derived shadow index（00 §8.2）。
非独立 MemoryProvider — 由 HybridRetriever 组合。

Layer 1：Protocol 空壳（接口签名完整，可 import 可 mock）。
Layer 6：具体实现（ChromaVectorStore / Neo4jGraphStore / FaissVectorStore / ...）。

INVARIANT:
- 叠加非互斥：Markdown truth（总在）+ optional VectorStore + optional GraphStore
  三层叠加，可同时启用
- adapter 是 Retriever 子组件：VectorStore/GraphStore 非独立 MemoryProvider，
  由 HybridRetriever 组合
- adapter 加载失败 → HybridRetriever no-op 跳过该路召回 + log warning（系统不崩）
"""
