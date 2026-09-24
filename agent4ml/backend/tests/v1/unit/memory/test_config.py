from __future__ import annotations

import dataclasses
import threading

import pytest
from dataclasses import replace

from agent4ml.backend.agents.memory.config import (
    STARTUP_ONLY_FIELDS,
    MemoryConfig,
    get_memory_config,
    set_memory_config,
)


@pytest.fixture(autouse=True)
def _reset_config() -> None:
    """每个测试前后重置全局 config 为默认，避免测试间污染。"""
    original = get_memory_config()
    yield
    set_memory_config(original)


class TestMemoryConfigDefaults:
    def test_use_empty(self) -> None:
        assert MemoryConfig().use == ""

    def test_storage_path_default(self) -> None:
        assert MemoryConfig().storage_path == ".agent4ml/memory"

    def test_enable_recall_default_true(self) -> None:
        assert MemoryConfig().enable_recall is True

    def test_enable_extract_default_false(self) -> None:
        assert MemoryConfig().enable_extract is False

    def test_token_budget_default(self) -> None:
        assert MemoryConfig().token_budget == 2000

    def test_vector_store_default_empty(self) -> None:
        assert MemoryConfig().vector_store == ""

    def test_graph_store_default_empty(self) -> None:
        assert MemoryConfig().graph_store == ""


class TestMemoryConfigDecayForgetPhase2:
    def test_decay_has_three_types(self) -> None:
        decay = MemoryConfig().decay
        assert "episodic" in decay
        assert "semantic" in decay
        assert "procedural" in decay

    def test_decay_episodic_params(self) -> None:
        decay = MemoryConfig().decay
        assert decay["episodic"] == {"base_strength": 0.7, "decay_rate": 0.1}

    def test_decay_semantic_params(self) -> None:
        decay = MemoryConfig().decay
        assert decay["semantic"] == {"base_strength": 0.8, "decay_rate": 0.02}

    def test_decay_procedural_params(self) -> None:
        decay = MemoryConfig().decay
        assert decay["procedural"] == {"base_strength": 0.9, "decay_rate": 0.005}

    def test_forget_defaults(self) -> None:
        forget = MemoryConfig().forget
        assert forget["strength_threshold"] == 0.1
        assert forget["ttl_hours"] == 720

    def test_phase2_defaults(self) -> None:
        phase2 = MemoryConfig().phase2
        assert phase2["enabled"] is False
        assert phase2["trigger_every_n_turns"] == 10
        assert phase2["trigger_on_session_end"] is True

    def test_decay_default_is_independent_per_instance(self) -> None:
        """frozen + default_factory：每个实例 decay dict 独立，不共享。"""
        c1 = MemoryConfig()
        c2 = MemoryConfig()
        assert c1.decay == c2.decay
        assert c1.decay is not c2.decay


class TestMemoryConfigFrozen:
    def test_top_level_frozen(self) -> None:
        config = MemoryConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.use = "default"

    def test_storage_path_frozen(self) -> None:
        config = MemoryConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.token_budget = 999


class TestStartupOnlyFields:
    def test_contains_use(self) -> None:
        assert "use" in STARTUP_ONLY_FIELDS

    def test_contains_storage_path(self) -> None:
        assert "storage_path" in STARTUP_ONLY_FIELDS

    def test_contains_vector_store(self) -> None:
        assert "vector_store" in STARTUP_ONLY_FIELDS

    def test_contains_graph_store(self) -> None:
        assert "graph_store" in STARTUP_ONLY_FIELDS

    def test_excludes_runtime_fields(self) -> None:
        assert "enable_recall" not in STARTUP_ONLY_FIELDS
        assert "token_budget" not in STARTUP_ONLY_FIELDS
        assert "decay" not in STARTUP_ONLY_FIELDS

    def test_exactly_four_fields(self) -> None:
        assert len(STARTUP_ONLY_FIELDS) == 4


class TestGetMemoryConfig:
    def test_returns_default_when_unset(self) -> None:
        """全局默认是 MemoryConfig()。"""
        config = get_memory_config()
        assert config.use == ""
        assert config.enable_recall is True


class TestSetMemoryConfig:
    def test_replace_takes_effect(self) -> None:
        new_config = replace(get_memory_config(), use="default")
        set_memory_config(new_config)
        assert get_memory_config().use == "default"

    def test_replace_does_not_mutate_original_object(self) -> None:
        """set_memory_config 整替全局引用，不影响原对象。"""
        original = get_memory_config()
        new_config = replace(original, use="default")
        set_memory_config(new_config)
        assert original.use == ""
        assert get_memory_config().use == "default"

    def test_thread_safety_under_concurrent_set(self) -> None:
        """多线程并发 set + get，不抛异常（_config_lock 保护）。"""
        results: list[str] = []
        errors: list[Exception] = []

        def writer() -> None:
            try:
                for i in range(50):
                    set_memory_config(replace(MemoryConfig(), use=f"u{i}"))
            except Exception as exc:
                errors.append(exc)

        def reader() -> None:
            try:
                for _ in range(50):
                    results.append(get_memory_config().use)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer) for _ in range(3)]
        threads += [threading.Thread(target=reader) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert len(results) == 150  # 3 reader × 50 次
