"""HybridRetriever — 纯 BM25 检索（无 vector/graph，48 §8.2）。

承接 `Hezao-MemDesign-Docs/agent4ml/50-memory-l3-store-retriever.md` §4 Step 2。

实现 Retriever Protocol。BM25 + retrieve 强化写回（1A）+ forgotten 过滤（3B）+ 增量索引（5B）。
retrieve 强化：调 decay_policy.compute_strength + trace.with_strength + store.update（1A 内部写回）。

INVARIANT：
- 纯 BM25，无 vector/graph 依赖
- forgotten 过滤在 Retriever（3B，store 不感知）
- 增量索引（5B）：构造全量建 + on_trace_* 增量维护
- retrieve 强化写回（1A）：命中后 store.update，caller 不负责
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from typing import Callable

from agent4ml.backend.agents.memory.memory_store import MemoryStore
from agent4ml.backend.agents.memory.retriever import Retriever
from agent4ml.backend.agents.memory.schema import MemoryTrace
from agent4ml.backend.agents.memory.strategies.default._constants import BM25_PARAMS
from agent4ml.backend.agents.memory.strategies.default.decay import EbbinghausDecayPolicy
from agent4ml.backend.agents.memory.types import MemoryQuery, RetrievalResult

logger = logging.getLogger(__name__)


class HybridRetriever:
    """纯 BM25 检索器（无 vector/graph）。

    构造时全量建索引（5B 冷启动）+ store 变更时增量维护（5B on_trace_*）。
    retrieve 强化写回（1A）：命中后 store.update。
    forgotten 过滤（3B）：retrieve 时排除 metadata.forgotten=True。
    """

    def __init__(
        self,
        store: MemoryStore,
        decay_policy: EbbinghausDecayPolicy,
        *,
        tokenize: Callable[[str], list[str]] | None = None,
    ) -> None:
        """初始化。

        Args:
            store: 持久化后端（MarkdownFileStore）
            decay_policy: 衰减策略（retrieve 强化时算 strength）
            tokenize: 分词函数（None 时默认空格分词，中文需 jieba 留后续）
        """
        self._store = store
        self._decay_policy = decay_policy
        self._tokenize = tokenize or self._default_tokenize
        # BM25 倒排索引：token → {trace_id: tf}
        self._inverted_index: dict[str, dict[str, int]] = defaultdict(dict)
        # trace 长度（分词后 token 数，BM25 用）
        self._trace_lengths: dict[str, int] = {}
        # 全量建索引（5B 冷启动）
        self._build_index_from_store()

    def _build_index_from_store(self) -> None:
        """5B 冷启动：从 store.list_all() 全量建 BM25 索引（排除 forgotten）。"""
        for trace in self._store.list_all():
            if trace.metadata.get("forgotten"):  # 3B forgotten 不入索引
                continue
            self._index_trace(trace)

    def _index_trace(self, trace: MemoryTrace) -> None:
        """单条 trace 入索引（5B 增量）。"""
        tokens = self._tokenize(trace.content)
        self._trace_lengths[trace.id] = len(tokens)
        tf: dict[str, int] = defaultdict(int)
        for token in tokens:
            tf[token] += 1
        for token, count in tf.items():
            self._inverted_index[token][trace.id] = count

    def _remove_trace_from_index(self, trace_id: str) -> None:
        """单条 trace 从索引移除（5B 增量）。"""
        self._trace_lengths.pop(trace_id, None)
        for token in list(self._inverted_index.keys()):
            self._inverted_index[token].pop(trace_id, None)
            if not self._inverted_index[token]:
                del self._inverted_index[token]

    def retrieve(self, query: MemoryQuery) -> list[RetrievalResult]:
        """检索相关记忆，返回按 score 降序排列的结果。

        1A：命中后调 decay_policy.compute_strength + trace.with_strength + store.update（强化写回）。
        3B：forgotten trace 不召回（filter metadata.forgotten != True）。
        复合分数：score = similarity × 0.7 + strength × 0.3。
        """
        now = time.time()
        query_tokens = self._tokenize(query.text)

        # 3B forgotten 过滤 + 候选集
        candidates = [t for t in self._store.list_all() if not t.metadata.get("forgotten")]
        if query.type_filter is not None:
            type_key = (
                query.type_filter.value
                if isinstance(query.type_filter, type(query.type_filter))
                else str(query.type_filter)
            )
            candidates = [t for t in candidates if t.type.value == type_key]

        # BM25 算分
        scores: list[tuple[MemoryTrace, float]] = []
        avgdl = sum(self._trace_lengths.values()) / max(1, len(self._trace_lengths))
        k1 = BM25_PARAMS["k1"]
        b = BM25_PARAMS["b"]

        for trace in candidates:
            similarity = self._bm25_score(query_tokens, trace.id, avgdl, k1, b)
            if similarity <= 0:
                continue
            # lazy decay 算 strength（1A 强化前先算当前值）
            current_strength = self._decay_policy.compute_strength(trace, now)
            if current_strength < query.min_strength:
                continue
            # 复合分数：score = similarity × 0.7 + strength × 0.3
            result = RetrievalResult.compute_score(trace, similarity, current_strength)
            scores.append((trace, result.score))

        # 按 score 降序 + top_k 截断
        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[: query.top_k]

        # 1A 强化写回：命中的 trace 调 with_strength + store.update
        results: list[RetrievalResult] = []
        for trace, score in top:
            new_strength = self._decay_policy.compute_strength(trace, now)
            strengthened = trace.with_strength(new_strength, now)
            self._store.update(strengthened)  # 1A 内部写回
            results.append(RetrievalResult.compute_score(strengthened, score, new_strength))

        return results

    def _bm25_score(
        self, query_tokens: list[str], trace_id: str, avgdl: float, k1: float, b: float
    ) -> float:
        """BM25 算分（标准公式）。"""
        score = 0.0
        trace_len = self._trace_lengths.get(trace_id, 0)
        if trace_len == 0:
            return 0.0
        for token in query_tokens:
            postings = self._inverted_index.get(token, {})
            tf = postings.get(trace_id, 0)
            if tf == 0:
                continue
            # IDF（+1 平滑防负）
            df = len(postings)
            n = len(self._trace_lengths)
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
            # TF 归一
            tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * trace_len / avgdl))
            score += idf * tf_norm
        return score

    @staticmethod
    def _default_tokenize(text: str) -> list[str]:
        """默认分词（空格 + 小写，中文需 Layer 3 换 jieba 留后续）。"""
        return text.lower().split()

    # 5B 增量索引接口（供 Layer 4 store 包装调用）
    def on_trace_added(self, trace: MemoryTrace) -> None:
        """store.add 后调（5B 增量）。"""
        if not trace.metadata.get("forgotten"):
            self._index_trace(trace)

    def on_trace_updated(self, trace: MemoryTrace) -> None:
        """store.update 后调（5B 增量，content 变了要重建）。"""
        self._remove_trace_from_index(trace.id)
        if not trace.metadata.get("forgotten"):
            self._index_trace(trace)

    def on_trace_removed(self, trace_id: str) -> None:
        """store.remove 后调（5B 增量）。"""
        self._remove_trace_from_index(trace_id)
