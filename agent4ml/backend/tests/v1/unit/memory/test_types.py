from __future__ import annotations

import dataclasses

import pytest

from agent4ml.backend.agents.memory.schema import MemoryTrace, MemoryType
from agent4ml.backend.agents.memory.types import MemoryFilter, MemoryQuery, RetrievalResult


def _make_trace() -> MemoryTrace:
    return MemoryTrace(id="t1", content="hello", type=MemoryType.EPISODIC)


class TestMemoryQuery:
    def test_required_text(self) -> None:
        q = MemoryQuery(text="query")
        assert q.text == "query"

    def test_default_top_k(self) -> None:
        assert MemoryQuery(text="q").top_k == 5

    def test_default_type_filter_none(self) -> None:
        assert MemoryQuery(text="q").type_filter is None

    def test_default_min_strength_zero(self) -> None:
        assert MemoryQuery(text="q").min_strength == 0.0

    def test_default_metadata_filter_empty(self) -> None:
        assert MemoryQuery(text="q").metadata_filter == {}

    def test_custom_fields(self) -> None:
        q = MemoryQuery(
            text="q",
            top_k=10,
            type_filter=MemoryType.SEMANTIC,
            min_strength=0.5,
            metadata_filter={"project": "agent4ml"},
        )
        assert q.top_k == 10
        assert q.type_filter is MemoryType.SEMANTIC
        assert q.min_strength == 0.5
        assert q.metadata_filter == {"project": "agent4ml"}

    def test_frozen(self) -> None:
        q = MemoryQuery(text="q")
        with pytest.raises(dataclasses.FrozenInstanceError):
            q.top_k = 10


class TestMemoryFilter:
    def test_default_type_filter_none(self) -> None:
        assert MemoryFilter().type_filter is None

    def test_default_min_strength_zero(self) -> None:
        assert MemoryFilter().min_strength == 0.0

    def test_default_max_age_hours_none(self) -> None:
        assert MemoryFilter().max_age_hours is None

    def test_default_metadata_filter_empty(self) -> None:
        assert MemoryFilter().metadata_filter == {}

    def test_custom_fields(self) -> None:
        f = MemoryFilter(
            type_filter=MemoryType.EPISODIC,
            min_strength=0.3,
            max_age_hours=24.0,
            metadata_filter={"tag": "travel"},
        )
        assert f.type_filter is MemoryType.EPISODIC
        assert f.min_strength == 0.3
        assert f.max_age_hours == 24.0
        assert f.metadata_filter == {"tag": "travel"}

    def test_frozen(self) -> None:
        f = MemoryFilter()
        with pytest.raises(dataclasses.FrozenInstanceError):
            f.min_strength = 0.5


class TestRetrievalResultComputeScore:
    def test_score_formula_0_8_0_6(self) -> None:
        """similarity=0.8, strength=0.6 → score=0.74（48 文档验收标准）。"""
        trace = _make_trace()
        result = RetrievalResult.compute_score(trace, similarity=0.8, strength=0.6)
        assert result.score == pytest.approx(0.74)
        assert result.similarity == 0.8
        assert result.strength == 0.6
        assert result.trace is trace

    def test_score_formula_zero_zero(self) -> None:
        result = RetrievalResult.compute_score(_make_trace(), 0.0, 0.0)
        assert result.score == 0.0

    def test_score_formula_one_one(self) -> None:
        result = RetrievalResult.compute_score(_make_trace(), 1.0, 1.0)
        assert result.score == 1.0

    def test_similarity_dominates_70_percent(self) -> None:
        """similarity 权重 0.7，strength 权重 0.3。"""
        result = RetrievalResult.compute_score(_make_trace(), similarity=1.0, strength=0.0)
        assert result.score == pytest.approx(0.7)
        result = RetrievalResult.compute_score(_make_trace(), similarity=0.0, strength=1.0)
        assert result.score == pytest.approx(0.3)

    def test_frozen(self) -> None:
        result = RetrievalResult.compute_score(_make_trace(), 0.5, 0.5)
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.score = 0.99


class TestRetrievalResultDirectConstruct:
    def test_direct_construct_with_explicit_score(self) -> None:
        """直接构造允许显式传 score（compute_score 是便捷 classmethod）。"""
        trace = _make_trace()
        result = RetrievalResult(trace=trace, similarity=0.5, strength=0.5, score=0.99)
        assert result.score == 0.99
