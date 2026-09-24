from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from agent4ml.backend.agents.leader.factory import _build_middlewares
from agent4ml.backend.agents.memory.bootstrap import _wrap_store, reset_memory_provider
from agent4ml.backend.agents.memory.config import get_memory_config, set_memory_config
from agent4ml.backend.agents.middlewares.memory_recall_middleware import MemoryMiddleware
from agent4ml.backend.agents.memory.schema import MemoryType
from agent4ml.backend.agents.memory.strategies.default.strategy import build_default_provider


@pytest.fixture(autouse=True)
def _reset():
    reset_memory_provider()
    original = get_memory_config()
    yield
    set_memory_config(original)
    reset_memory_provider()


class TestE2eRecallWithRealStore:
    def test_recall_injects_and_strengthens(self, tmp_path: Path) -> None:
        """端到端：build_default_provider + _wrap_store + encode + abefore_model recall + 强化写回。"""
        config = replace(get_memory_config(), use="default", storage_path=str(tmp_path))
        set_memory_config(config)

        provider = build_default_provider()
        _wrap_store(provider.store(), provider.retriever())

        manager = provider.manager()
        trace = manager.encode("tokyo travel plan", MemoryType.EPISODIC)
        assert trace.id

        mw = MemoryMiddleware(provider)
        state = {"messages": [HumanMessage(content="tokyo")]}

        patch = asyncio.run(mw.abefore_model(state, runtime=None))
        assert patch is not None
        assert len(patch["messages"]) == 1
        assert patch["messages"][0].name == "memory_recall"
        recalled = patch["recalled_memories"]
        assert len(recalled) >= 1
        assert any(r["id"] == trace.id for r in recalled)

        # 强化写回 1A：access_count+1
        updated_trace = provider.store().get(trace.id)
        assert updated_trace.access_count == 1

    def test_no_match_returns_none(self, tmp_path: Path) -> None:
        """retrieve 无匹配时返 None。"""
        config = replace(get_memory_config(), use="default", storage_path=str(tmp_path))
        set_memory_config(config)

        provider = build_default_provider()
        _wrap_store(provider.store(), provider.retriever())
        manager = provider.manager()
        manager.encode("cooking recipe", MemoryType.EPISODIC)

        mw = MemoryMiddleware(provider)
        state = {"messages": [HumanMessage(content="tokyo travel")]}

        patch = asyncio.run(mw.abefore_model(state, runtime=None))
        assert patch is None


class TestE2eNoMemoryProvider:
    def test_no_provider_no_middleware(self) -> None:
        """memory_provider=None 不挂载 MemoryMiddleware（向后兼容）。"""
        middlewares = _build_middlewares(expert_mode=False, memory_provider=None)
        assert not any(isinstance(m, MemoryMiddleware) for m in middlewares)
