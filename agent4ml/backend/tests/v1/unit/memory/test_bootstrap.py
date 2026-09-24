from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from agent4ml.backend.agents.memory.bootstrap import (
    _make_journal_callback,
    _wrap_store,
    get_memory_provider,
    reset_memory_provider,
    set_memory_provider,
    shutdown_memory_provider,
)
from agent4ml.backend.agents.memory.config import get_memory_config, set_memory_config


@pytest.fixture(autouse=True)
def _reset_provider():
    """每个测试前后重置全局 provider，避免测试间污染。"""
    reset_memory_provider()
    yield
    reset_memory_provider()


@pytest.fixture(autouse=True)
def _reset_config():
    original = get_memory_config()
    yield
    set_memory_config(original)


class TestGetMemoryProvider:
    def test_config_use_empty_returns_none(self) -> None:
        """config.use="" 返 None（记忆禁用）。"""
        config = replace(get_memory_config(), use="")
        set_memory_config(config)
        assert get_memory_provider() is None

    def test_config_use_default_returns_provider(self, tmp_path: Path) -> None:
        """config.use="default" 返 provider（L3 build_default_provider 实例化）。"""
        config = replace(get_memory_config(), use="default", storage_path=str(tmp_path))
        set_memory_config(config)
        provider = get_memory_provider()
        assert provider is not None

    def test_lazy_load_caches(self, tmp_path: Path) -> None:
        """懒加载：第二次调用返同一实例（不重复加载）。"""
        config = replace(get_memory_config(), use="default", storage_path=str(tmp_path))
        set_memory_config(config)
        p1 = get_memory_provider()
        p2 = get_memory_provider()
        assert p1 is p2


class TestResetMemoryProvider:
    def test_reset_clears_cache(self, tmp_path: Path) -> None:
        config = replace(get_memory_config(), use="default", storage_path=str(tmp_path))
        set_memory_config(config)
        get_memory_provider()  # 加载
        reset_memory_provider()
        # reset 后再 get 应重新加载（不同实例）
        # 但 set_memory_config 不变，storage_path 同，可能返同实例（MarkdownFileStore 同路径）
        # 验证 reset 后 _memory_provider 是 None（内部状态）
        from agent4ml.backend.agents.memory import bootstrap
        assert bootstrap._memory_provider is None


class TestShutdownMemoryProvider:
    def test_shutdown_clears_cache(self, tmp_path: Path) -> None:
        config = replace(get_memory_config(), use="default", storage_path=str(tmp_path))
        set_memory_config(config)
        get_memory_provider()  # 加载
        shutdown_memory_provider()
        from agent4ml.backend.agents.memory import bootstrap
        assert bootstrap._memory_provider is None

    def test_shutdown_calls_provider_shutdown(self) -> None:
        """hasattr duck-type：provider 有 shutdown 时调用。"""

        class _ProviderWithShutdown:
            def __init__(self):
                self.shutdown_called = False

            def shutdown(self):
                self.shutdown_called = True

            def store(self):
                return None

            def retriever(self):
                return None

            def manager(self):
                return None

        provider = _ProviderWithShutdown()
        set_memory_provider(provider)
        shutdown_memory_provider()
        assert provider.shutdown_called is True

    def test_shutdown_silent_when_no_shutdown_method(self) -> None:
        """provider 无 shutdown 时静默（hasattr duck-type）。"""

        class _ProviderNoShutdown:
            def store(self):
                return None

            def retriever(self):
                return None

            def manager(self):
                return None

        provider = _ProviderNoShutdown()
        set_memory_provider(provider)
        # 不抛错
        shutdown_memory_provider()


class TestSetMemoryProvider:
    def test_set_injects_provider(self) -> None:
        """set 注入后 get 返注入的 provider。"""
        mock = object()
        set_memory_provider(mock)
        assert get_memory_provider() is mock

    def test_set_overrides_config(self) -> None:
        """set 注入优先于 config 反射加载。"""
        config = replace(get_memory_config(), use="default")
        set_memory_config(config)
        mock = object()
        set_memory_provider(mock)
        assert get_memory_provider() is mock


class TestMakeJournalCallback:
    def test_returns_callable_when_run_journal_available(self, monkeypatch) -> None:
        """RunJournal 可用时返 lambda。"""

        class _MockJournal:
            def __init__(self):
                self.events = []

            def append(self, event, payload):
                self.events.append((event, payload))

        journal = _MockJournal()
        monkeypatch.setattr(
            "agent4ml.backend.agents.journal.get_run_journal", lambda: journal, raising=False,
        )
        callback = _make_journal_callback()
        assert callback is not None
        assert callable(callback)
        callback("memory.test", {"k": "v"})
        assert journal.events == [("memory.test", {"k": "v"})]

    def test_returns_none_when_run_journal_unavailable(self, monkeypatch) -> None:
        """RunJournal 不可用时返 None。"""
        monkeypatch.setattr(
            "agent4ml.backend.agents.journal.get_run_journal",
            lambda: (_ for _ in ()).throw(ImportError("no journal")),
            raising=False,
        )
        callback = _make_journal_callback()
        assert callback is None

    def test_returns_none_when_get_run_journal_returns_none(self, monkeypatch) -> None:
        """get_run_journal() 返 None 时返 None。"""
        monkeypatch.setattr("agent4ml.backend.agents.journal.get_run_journal", lambda: None, raising=False)
        callback = _make_journal_callback()
        assert callback is None


class _MockStoreForWrap:
    def __init__(self):
        self.add_calls = []
        self.update_calls = []
        self.batch_update_calls = []
        self.remove_calls = []

    def add(self, trace):
        self.add_calls.append(trace)

    def update(self, trace):
        self.update_calls.append(trace)

    def batch_update(self, traces):
        self.batch_update_calls.append(list(traces))

    def remove(self, trace_id):
        self.remove_calls.append(trace_id)


class _MockRetrieverForWrap:
    def __init__(self):
        self.on_added = []
        self.on_updated = []
        self.on_removed = []

    def on_trace_added(self, trace):
        self.on_added.append(trace)

    def on_trace_updated(self, trace):
        self.on_updated.append(trace)

    def on_trace_removed(self, trace_id):
        self.on_removed.append(trace_id)


class TestWrapStore:
    def test_add_calls_on_trace_added(self) -> None:
        store = _MockStoreForWrap()
        retriever = _MockRetrieverForWrap()
        _wrap_store(store, retriever)
        trace = object()
        store.add(trace)
        assert store.add_calls == [trace]
        assert retriever.on_added == [trace]

    def test_update_calls_on_trace_updated(self) -> None:
        store = _MockStoreForWrap()
        retriever = _MockRetrieverForWrap()
        _wrap_store(store, retriever)
        trace = object()
        store.update(trace)
        assert store.update_calls == [trace]
        assert retriever.on_updated == [trace]

    def test_batch_update_calls_on_trace_updated_per_trace(self) -> None:
        store = _MockStoreForWrap()
        retriever = _MockRetrieverForWrap()
        _wrap_store(store, retriever)
        t1, t2, t3 = object(), object(), object()
        store.batch_update([t1, t2, t3])
        assert store.batch_update_calls == [[t1, t2, t3]]
        assert retriever.on_updated == [t1, t2, t3]

    def test_remove_calls_on_trace_removed(self) -> None:
        store = _MockStoreForWrap()
        retriever = _MockRetrieverForWrap()
        _wrap_store(store, retriever)
        store.remove("trace-id-123")
        assert store.remove_calls == ["trace-id-123"]
        assert retriever.on_removed == ["trace-id-123"]

    def test_wrapped_methods_replace_originals(self) -> None:
        store = _MockStoreForWrap()
        retriever = _MockRetrieverForWrap()
        original_add = store.add
        _wrap_store(store, retriever)
        assert store.add is not original_add  # 包装后是 wrapped 版本
