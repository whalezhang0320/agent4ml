from __future__ import annotations

import dataclasses

import pytest

from agent4ml.backend.agents.memory.memory_provider import MemoryProvider
from agent4ml.backend.agents.memory.schema import MemoryTrace, MemoryType
from agent4ml.backend.agents.memory.strategies.default.decay import EbbinghausDecayPolicy
from agent4ml.backend.agents.memory.strategies.default.forget import CompositeForgetPolicy
from agent4ml.backend.agents.memory.strategies.default.manager import DefaultMemoryManager
from agent4ml.backend.agents.memory.strategies.default.strategy import (
    DefaultMemoryProvider,
    build_default_provider,
)
from agent4ml.backend.agents.memory.types import MemoryFilter, MemoryQuery, RetrievalResult


class _MockStore:
    def add(self, trace: MemoryTrace) -> None: ...
    def get(self, trace_id: str) -> MemoryTrace | None: return None
    def update(self, trace: MemoryTrace) -> None: ...
    def batch_update(self, traces: list[MemoryTrace]) -> None: ...
    def remove(self, trace_id: str) -> None: ...
    def list_by_type(self, type: MemoryType) -> list[MemoryTrace]: return []
    def list_by_filter(self, filter: MemoryFilter) -> list[MemoryTrace]: return []
    def list_all(self) -> list[MemoryTrace]: return []


class _MockRetriever:
    def retrieve(self, query: MemoryQuery) -> list[RetrievalResult]: return []


class _MockStoreWithShutdown(_MockStore):
    def __init__(self) -> None:
        self.shutdown_called = False

    def shutdown(self) -> None:
        self.shutdown_called = True


class _MockRetrieverWithShutdown(_MockRetriever):
    def __init__(self) -> None:
        self.shutdown_called = False

    def shutdown(self) -> None:
        self.shutdown_called = True


class TestBuildDefaultProvider:
    def test_returns_default_memory_provider(self) -> None:
        store = _MockStore()
        retriever = _MockRetriever()
        provider = build_default_provider(store=store, retriever=retriever)
        assert isinstance(provider, DefaultMemoryProvider)

    def test_returns_memory_provider_protocol(self) -> None:
        store = _MockStore()
        retriever = _MockRetriever()
        provider = build_default_provider(store=store, retriever=retriever)
        assert isinstance(provider, MemoryProvider)

    def test_three_components_non_none(self) -> None:
        store = _MockStore()
        retriever = _MockRetriever()
        provider = build_default_provider(store=store, retriever=retriever)
        assert provider.store() is store
        assert provider.retriever() is retriever
        assert provider.manager() is not None

    def test_manager_is_default_memory_manager(self) -> None:
        store = _MockStore()
        retriever = _MockRetriever()
        provider = build_default_provider(store=store, retriever=retriever)
        assert isinstance(provider.manager(), DefaultMemoryManager)

    def test_default_strategies_injected(self) -> None:
        store = _MockStore()
        retriever = _MockRetriever()
        provider = build_default_provider(store=store, retriever=retriever)
        manager = provider.manager()
        assert isinstance(manager._decay_policy, EbbinghausDecayPolicy)
        assert isinstance(manager._forget_policy, CompositeForgetPolicy)

    def test_custom_decay_policy_injected(self) -> None:
        store = _MockStore()
        retriever = _MockRetriever()
        custom_decay = EbbinghausDecayPolicy()
        provider = build_default_provider(
            store=store, retriever=retriever, decay_policy=custom_decay
        )
        assert provider.manager()._decay_policy is custom_decay

    def test_custom_forget_policy_injected(self) -> None:
        store = _MockStore()
        retriever = _MockRetriever()
        custom_forget = CompositeForgetPolicy()
        provider = build_default_provider(
            store=store, retriever=retriever, forget_policy=custom_forget
        )
        assert provider.manager()._forget_policy is custom_forget

    def test_journal_passthrough(self) -> None:
        store = _MockStore()
        retriever = _MockRetriever()
        events: list[str] = []

        def journal(event: str, payload: dict) -> None:
            events.append(event)

        provider = build_default_provider(
            store=store, retriever=retriever, journal=journal
        )
        assert provider.manager()._journal is journal

    def test_journal_none_default(self) -> None:
        store = _MockStore()
        retriever = _MockRetriever()
        provider = build_default_provider(store=store, retriever=retriever)
        assert provider.manager()._journal is None

    def test_not_raises_not_implemented(self) -> None:
        store = _MockStore()
        retriever = _MockRetriever()
        provider = build_default_provider(store=store, retriever=retriever)
        assert provider is not None


class TestDefaultMemoryProviderShutdown:
    def test_shutdown_calls_store_shutdown(self) -> None:
        store = _MockStoreWithShutdown()
        retriever = _MockRetriever()
        provider = build_default_provider(store=store, retriever=retriever)
        provider.shutdown()
        assert store.shutdown_called is True

    def test_shutdown_calls_retriever_shutdown(self) -> None:
        store = _MockStore()
        retriever = _MockRetrieverWithShutdown()
        provider = build_default_provider(store=store, retriever=retriever)
        provider.shutdown()
        assert retriever.shutdown_called is True

    def test_shutdown_silent_when_no_shutdown_method(self) -> None:
        store = _MockStore()
        retriever = _MockRetriever()
        provider = build_default_provider(store=store, retriever=retriever)
        provider.shutdown()

    def test_shutdown_calls_both(self) -> None:
        store = _MockStoreWithShutdown()
        retriever = _MockRetrieverWithShutdown()
        provider = build_default_provider(store=store, retriever=retriever)
        provider.shutdown()
        assert store.shutdown_called is True
        assert retriever.shutdown_called is True


class TestDefaultMemoryProviderFrozen:
    def test_frozen_top_level(self) -> None:
        store = _MockStore()
        retriever = _MockRetriever()
        provider = build_default_provider(store=store, retriever=retriever)
        with pytest.raises(dataclasses.FrozenInstanceError):
            provider._store = _MockStore()  # type: ignore[misc]

    def test_is_dataclass(self) -> None:
        store = _MockStore()
        retriever = _MockRetriever()
        provider = build_default_provider(store=store, retriever=retriever)
        assert dataclasses.is_dataclass(provider)


class TestBuildDefaultProviderNoArgs:
    def test_no_args_instantiate_from_config(self, tmp_path) -> None:
        """Layer 3 完整版：无参时从 config 实例化 MarkdownFileStore + HybridRetriever。"""
        from dataclasses import replace

        from agent4ml.backend.agents.memory.config import get_memory_config, set_memory_config
        from agent4ml.backend.agents.memory.strategies.default.retriever import HybridRetriever
        from agent4ml.backend.agents.memory.strategies.default.store import MarkdownFileStore

        original = get_memory_config()
        try:
            new_config = replace(original, storage_path=str(tmp_path))
            set_memory_config(new_config)
            provider = build_default_provider()
            assert isinstance(provider.store(), MarkdownFileStore)
            assert isinstance(provider.retriever(), HybridRetriever)
            assert isinstance(provider.manager(), DefaultMemoryManager)
        finally:
            set_memory_config(original)
