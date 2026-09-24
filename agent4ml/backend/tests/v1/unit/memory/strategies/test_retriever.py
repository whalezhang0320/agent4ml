from __future__ import annotations

import time
from pathlib import Path

import pytest

from agent4ml.backend.agents.memory.schema import MemoryTrace, MemoryType
from agent4ml.backend.agents.memory.strategies.default.decay import EbbinghausDecayPolicy
from agent4ml.backend.agents.memory.strategies.default.retriever import HybridRetriever
from agent4ml.backend.agents.memory.strategies.default.store import MarkdownFileStore
from agent4ml.backend.agents.memory.types import MemoryQuery


def _make_trace(
    *,
    id: str,
    content: str,
    type: MemoryType = MemoryType.EPISODIC,
    last_accessed: float = 0.0,
    created_at: float = 0.0,
    metadata: dict | None = None,
) -> MemoryTrace:
    return MemoryTrace(
        id=id, content=content, type=type,
        last_accessed=last_accessed, created_at=created_at or time.time(),
        metadata=metadata or {},
    )


def _make_retriever(tmp_path: Path) -> tuple[MarkdownFileStore, HybridRetriever]:
    store = MarkdownFileStore(tmp_path)
    decay = EbbinghausDecayPolicy()
    retriever = HybridRetriever(store, decay)
    return store, retriever


class TestRetrieveBm25:
    def test_retrieve_returns_relevant(self, tmp_path: Path) -> None:
        """BM25 召回相关 trace。"""
        store, retriever = _make_retriever(tmp_path)
        store.add(_make_trace(id="aaaa1111aaaa1111", content="tokyo travel plan"))
        store.add(_make_trace(id="bbbb2222bbbb2222", content="cooking recipe pasta"))
        # retriever 冷启动已建索引（构造时），但 add 在构造后，需手动 on_trace_added
        retriever.on_trace_added(store.get("aaaa1111aaaa1111"))
        retriever.on_trace_added(store.get("bbbb2222bbbb2222"))

        results = retriever.retrieve(MemoryQuery(text="tokyo", top_k=5))
        ids = [r.trace.id for r in results]
        assert "aaaa1111aaaa1111" in ids
        assert "bbbb2222bbbb2222" not in ids

    def test_retrieve_top_k_truncation(self, tmp_path: Path) -> None:
        """top_k 截断。"""
        store, retriever = _make_retriever(tmp_path)
        for i in range(5):
            tid = f"{i:04x}0000000000000"[:16]
            store.add(_make_trace(id=tid, content=f"tokyo travel {i}"))
            retriever.on_trace_added(store.get(tid))
        results = retriever.retrieve(MemoryQuery(text="tokyo", top_k=2))
        assert len(results) <= 2

    def test_bm25_more_hits_higher_score(self, tmp_path: Path) -> None:
        """查询词命中多分数高。"""
        store, retriever = _make_retriever(tmp_path)
        store.add(_make_trace(id="aaaa1111aaaa1111", content="tokyo travel"))
        store.add(_make_trace(id="bbbb2222bbbb2222", content="tokyo"))
        retriever.on_trace_added(store.get("aaaa1111aaaa1111"))
        retriever.on_trace_added(store.get("bbbb2222bbbb2222"))
        results = retriever.retrieve(MemoryQuery(text="tokyo travel", top_k=5))
        # aaaa（2 命中）应排前
        assert results[0].trace.id == "aaaa1111aaaa1111"


class TestRetrieveStrengthenWriteback:
    def test_strengthen_updates_access_count(self, tmp_path: Path) -> None:
        """1A 强化写回：命中 trace access_count+1 + store.update。"""
        store, retriever = _make_retriever(tmp_path)
        store.add(_make_trace(id="aaaa1111aaaa1111", content="tokyo travel"))
        retriever.on_trace_added(store.get("aaaa1111aaaa1111"))
        before = store.get("aaaa1111aaaa1111").access_count

        retriever.retrieve(MemoryQuery(text="tokyo", top_k=5))
        after = store.get("aaaa1111aaaa1111").access_count
        assert after == before + 1

    def test_strengthen_updates_last_accessed(self, tmp_path: Path) -> None:
        """1A 强化写回：last_accessed 更新。"""
        store, retriever = _make_retriever(tmp_path)
        store.add(_make_trace(id="aaaa1111aaaa1111", content="tokyo", last_accessed=1.0))
        retriever.on_trace_added(store.get("aaaa1111aaaa1111"))

        retriever.retrieve(MemoryQuery(text="tokyo", top_k=5))
        after = store.get("aaaa1111aaaa1111").last_accessed
        assert after > 1.0


class TestForgottenFilter:
    def test_forgotten_not_retrieved(self, tmp_path: Path) -> None:
        """3B forgotten 过滤：forgotten trace 不召回。"""
        store, retriever = _make_retriever(tmp_path)
        store.add(_make_trace(id="aaaa1111aaaa1111", content="tokyo", metadata={"forgotten": True}))
        store.add(_make_trace(id="bbbb2222bbbb2222", content="tokyo"))
        retriever.on_trace_added(store.get("aaaa1111aaaa1111"))
        retriever.on_trace_added(store.get("bbbb2222bbbb2222"))

        results = retriever.retrieve(MemoryQuery(text="tokyo", top_k=5))
        ids = [r.trace.id for r in results]
        assert "aaaa1111aaaa1111" not in ids
        assert "bbbb2222bbbb2222" in ids


class TestRetrieveFilters:
    def test_type_filter(self, tmp_path: Path) -> None:
        store, retriever = _make_retriever(tmp_path)
        store.add(_make_trace(id="aaaa1111aaaa1111", content="tokyo", type=MemoryType.EPISODIC))
        store.add(_make_trace(id="bbbb2222bbbb2222", content="tokyo", type=MemoryType.SEMANTIC))
        retriever.on_trace_added(store.get("aaaa1111aaaa1111"))
        retriever.on_trace_added(store.get("bbbb2222bbbb2222"))

        results = retriever.retrieve(MemoryQuery(text="tokyo", type_filter=MemoryType.SEMANTIC))
        assert len(results) == 1
        assert results[0].trace.type is MemoryType.SEMANTIC

    def test_min_strength_filter(self, tmp_path: Path) -> None:
        """min_strength 过滤：低 strength 不召回。"""
        store, retriever = _make_retriever(tmp_path)
        # 老 trace，衰减后 strength 很低
        store.add(_make_trace(id="aaaa1111aaaa1111", content="tokyo", last_accessed=1.0))
        retriever.on_trace_added(store.get("aaaa1111aaaa1111"))

        results = retriever.retrieve(MemoryQuery(text="tokyo", min_strength=0.9))
        # strength 衰减后 < 0.9，不召回
        assert len(results) == 0


class TestColdStart:
    def test_build_index_on_construction(self, tmp_path: Path) -> None:
        """5B 冷启动：构造时从 store.list_all 全量建索引。"""
        store = MarkdownFileStore(tmp_path)
        store.add(_make_trace(id="aaaa1111aaaa1111", content="tokyo travel"))
        # 构造 retriever（冷启动建索引），不需 on_trace_added
        retriever = HybridRetriever(store, EbbinghausDecayPolicy())
        results = retriever.retrieve(MemoryQuery(text="tokyo", top_k=5))
        assert len(results) == 1
        assert results[0].trace.id == "aaaa1111aaaa1111"

    def test_cold_start_excludes_forgotten(self, tmp_path: Path) -> None:
        """冷启动 forgotten 不入索引。"""
        store = MarkdownFileStore(tmp_path)
        store.add(_make_trace(id="aaaa1111aaaa1111", content="tokyo", metadata={"forgotten": True}))
        store.add(_make_trace(id="bbbb2222bbbb2222", content="tokyo"))
        retriever = HybridRetriever(store, EbbinghausDecayPolicy())
        results = retriever.retrieve(MemoryQuery(text="tokyo", top_k=5))
        ids = [r.trace.id for r in results]
        assert "aaaa1111aaaa1111" not in ids
        assert "bbbb2222bbbb2222" in ids


class TestIncrementalIndex:
    def test_on_trace_added(self, tmp_path: Path) -> None:
        """5B on_trace_added 增量入索引。"""
        store, retriever = _make_retriever(tmp_path)
        store.add(_make_trace(id="aaaa1111aaaa1111", content="tokyo"))
        retriever.on_trace_added(store.get("aaaa1111aaaa1111"))
        results = retriever.retrieve(MemoryQuery(text="tokyo", top_k=5))
        assert any(r.trace.id == "aaaa1111aaaa1111" for r in results)

    def test_on_trace_updated_rebuilds(self, tmp_path: Path) -> None:
        """5B on_trace_updated：content 变了重建索引。"""
        store, retriever = _make_retriever(tmp_path)
        store.add(_make_trace(id="aaaa1111aaaa1111", content="tokyo"))
        retriever.on_trace_added(store.get("aaaa1111aaaa1111"))
        # 更新 content
        updated = _make_trace(id="aaaa1111aaaa1111", content="osaka")
        store.update(updated)
        retriever.on_trace_updated(updated)
        # 旧 content "tokyo" 不再命中
        results_old = retriever.retrieve(MemoryQuery(text="tokyo", top_k=5))
        assert all(r.trace.id != "aaaa1111aaaa1111" for r in results_old)
        # 新 content "osaka" 命中
        results_new = retriever.retrieve(MemoryQuery(text="osaka", top_k=5))
        assert any(r.trace.id == "aaaa1111aaaa1111" for r in results_new)

    def test_on_trace_removed(self, tmp_path: Path) -> None:
        """5B on_trace_removed 从索引移除。"""
        store, retriever = _make_retriever(tmp_path)
        store.add(_make_trace(id="aaaa1111aaaa1111", content="tokyo"))
        retriever.on_trace_added(store.get("aaaa1111aaaa1111"))
        retriever.on_trace_removed("aaaa1111aaaa1111")
        results = retriever.retrieve(MemoryQuery(text="tokyo", top_k=5))
        assert all(r.trace.id != "aaaa1111aaaa1111" for r in results)

    def test_on_trace_updated_forgotten_removes(self, tmp_path: Path) -> None:
        """on_trace_updated forgotten 标记后从索引移除。"""
        store, retriever = _make_retriever(tmp_path)
        store.add(_make_trace(id="aaaa1111aaaa1111", content="tokyo"))
        retriever.on_trace_added(store.get("aaaa1111aaaa1111"))
        forgotten = _make_trace(id="aaaa1111aaaa1111", content="tokyo", metadata={"forgotten": True})
        retriever.on_trace_updated(forgotten)
        results = retriever.retrieve(MemoryQuery(text="tokyo", top_k=5))
        assert all(r.trace.id != "aaaa1111aaaa1111" for r in results)


class TestNoVectorGraph:
    def test_no_vector_graph_import(self) -> None:
        """无 vector/graph 依赖。"""
        import agent4ml.backend.agents.memory.strategies.default.retriever as mod
        names = [n.lower() for n in dir(mod)]
        assert "vectorstore" not in names
        assert "graphstore" not in names
        assert "chroma" not in names
        assert "faiss" not in names
        assert "neo4j" not in names
