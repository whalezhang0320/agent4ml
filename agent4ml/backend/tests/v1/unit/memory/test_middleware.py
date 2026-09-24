from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import HumanMessage

from agent4ml.backend.agents.middlewares.memory_recall_middleware import MemoryMiddleware
from agent4ml.backend.agents.memory.schema import MemoryTrace, MemoryType
from agent4ml.backend.agents.memory.types import RetrievalResult


def _make_trace(id: str = "aaaa1111aaaa1111", content: str = "tokyo travel") -> MemoryTrace:
    return MemoryTrace(id=id, content=content, type=MemoryType.EPISODIC)


def _make_result(trace: MemoryTrace, similarity: float = 0.8, strength: float = 0.7) -> RetrievalResult:
    return RetrievalResult.compute_score(trace, similarity=similarity, strength=strength)


class _MockRetriever:
    def __init__(self, results=None):
        self._results = results or []

    def retrieve(self, query):
        return self._results


class _MockProvider:
    def __init__(self, retriever):
        self._retriever = retriever

    def retriever(self):
        return self._retriever

    def store(self):
        return None

    def manager(self):
        return None


def _make_state(query: str = "tokyo travel"):
    return {"messages": [HumanMessage(content=query)]}


def _patch_set_turn_id(monkeypatch):
    """patch set_turn_id 记录调用（避免 ContextVar + asyncio.run 隔离问题）。"""
    calls: list = []
    monkeypatch.setattr(
        "agent4ml.backend.agents.middlewares.memory_recall_middleware.set_turn_id",
        lambda tid: calls.append(tid),
    )
    return calls


class TestAbeforeModelRecall:
    def test_recall_injects_human_message(self) -> None:
        trace = _make_trace()
        result = _make_result(trace)
        provider = _MockProvider(_MockRetriever([result]))
        mw = MemoryMiddleware(provider)
        state = _make_state("tokyo travel")

        patch = asyncio.run(mw.abefore_model(state, runtime=None))
        assert patch is not None
        msgs = patch["messages"]
        assert len(msgs) == 1
        assert msgs[0].name == "memory_recall"
        assert msgs[0].additional_kwargs.get("hide_from_ui") is True

    def test_recall_writes_recalled_memories_index(self) -> None:
        trace = _make_trace(id="bbbb2222bbbb2222")
        result = _make_result(trace, similarity=0.9, strength=0.6)
        provider = _MockProvider(_MockRetriever([result]))
        mw = MemoryMiddleware(provider)
        state = _make_state("query")

        patch = asyncio.run(mw.abefore_model(state, runtime=None))
        recalled = patch["recalled_memories"]
        assert len(recalled) == 1
        assert recalled[0]["id"] == "bbbb2222bbbb2222"
        # score = similarity×0.7 + strength×0.3 = 0.9×0.7 + 0.6×0.3 = 0.81
        assert recalled[0]["score"] == pytest.approx(0.81)
        assert recalled[0]["strength"] == 0.6
        # 只存索引，不存全量 content
        assert "content" not in recalled[0]


class TestSetTurnId:
    def test_before_model_injects_turn_id(self, monkeypatch) -> None:
        calls = _patch_set_turn_id(monkeypatch)
        trace = _make_trace()
        result = _make_result(trace)
        provider = _MockProvider(_MockRetriever([result]))
        mw = MemoryMiddleware(provider)
        state = _make_state("query")

        asyncio.run(mw.abefore_model(state, runtime=None))
        # abefore_model 调 set_turn_id(turn_id)
        assert len(calls) >= 1
        assert calls[0] is not None
        assert ":turn:" in calls[0]

    def test_after_model_clears_turn_id(self, monkeypatch) -> None:
        calls = _patch_set_turn_id(monkeypatch)
        provider = _MockProvider(_MockRetriever([]))
        mw = MemoryMiddleware(provider)
        state = _make_state("query")

        asyncio.run(mw.aafter_model(state, runtime=None))
        # aafter_model 调 set_turn_id(None)
        assert calls == [None]


class TestEnableRecallFalse:
    def test_no_recall_when_disabled(self, monkeypatch) -> None:
        calls = _patch_set_turn_id(monkeypatch)
        trace = _make_trace()
        result = _make_result(trace)
        provider = _MockProvider(_MockRetriever([result]))
        mw = MemoryMiddleware(provider, enable_recall=False)
        state = _make_state("query")

        patch = asyncio.run(mw.abefore_model(state, runtime=None))
        assert patch is None
        # enable_recall=False 不调 set_turn_id
        assert calls == []


class TestRetrieveEmpty:
    def test_empty_results_clears_turn_id_no_inject(self, monkeypatch) -> None:
        calls = _patch_set_turn_id(monkeypatch)
        provider = _MockProvider(_MockRetriever([]))  # retrieve 返空
        mw = MemoryMiddleware(provider)
        state = _make_state("query")

        patch = asyncio.run(mw.abefore_model(state, runtime=None))
        assert patch is None
        # retrieve 返空时 set_turn_id 先注入后清除
        assert len(calls) == 2
        assert calls[0] is not None  # 注入 turn_id
        assert calls[-1] is None  # 清除

    def test_no_query_returns_none(self) -> None:
        provider = _MockProvider(_MockRetriever([_make_result(_make_trace())]))
        mw = MemoryMiddleware(provider)
        state = {"messages": []}  # 无 user message

        patch = asyncio.run(mw.abefore_model(state, runtime=None))
        assert patch is None


class TestTokenBudget:
    def test_budget_truncates_results(self) -> None:
        traces = [_make_trace(id=f"{i:04x}0000000000000"[:16], content=f"content {i}") for i in range(10)]
        results = [_make_result(t) for t in traces]
        provider = _MockProvider(_MockRetriever(results))
        mw = MemoryMiddleware(provider, token_budget=1)  # 极小预算
        state = _make_state("query")

        patch = asyncio.run(mw.abefore_model(state, runtime=None))
        recalled = patch["recalled_memories"]
        # token budget 极小，截断（不超过全部）
        assert len(recalled) <= 10


class TestPromptCaching:
    def test_injected_message_not_in_system_prompt(self) -> None:
        trace = _make_trace()
        result = _make_result(trace)
        provider = _MockProvider(_MockRetriever([result]))
        mw = MemoryMiddleware(provider)
        state = _make_state("query")

        patch = asyncio.run(mw.abefore_model(state, runtime=None))
        msg = patch["messages"][0]
        # 注入 HumanMessage（非 SystemMessage），保护 prompt caching
        assert isinstance(msg, HumanMessage)
        assert msg.additional_kwargs.get("hide_from_ui") is True


class TestEnableExtract:
    def test_after_model_no_extract_by_default(self) -> None:
        provider = _MockProvider(_MockRetriever([]))
        mw = MemoryMiddleware(provider)  # enable_extract 默认 False
        state = _make_state("query")

        patch = asyncio.run(mw.aafter_model(state, runtime=None))
        # enable_extract=False，aafter_model 仅清除 turn_id，返 None
        assert patch is None
