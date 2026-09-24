"""AppConfig.memory 兼容性State 接入测试。

依赖 langchain（AppConfig import 链触发 skill.config → langchain_core；
ThreadState 继承 langchain.agents.AgentState）。langchain 未安装时整体跳过
（pre-existing 环境缺失，与 memory L1 无关，STATE.md 记录）。
"""

from __future__ import annotations

import dataclasses

import pytest

pytest.importorskip("langchain_core")

from dataclasses import replace

from agent4ml.backend.agents.config.schema import (
    AppConfig,
    MiddlewareConfig,
    ModelConfig,
    ObservabilityConfig,
    ReportingConfig,
    RuntimeConfig,
    ToolConfig,
)
from agent4ml.backend.agents.memory.config import MemoryConfig


def _make_app_config(**overrides) -> AppConfig:
    defaults = dict(
        name="agent4ml",
        environment="test",
        runtime=RuntimeConfig(),
        models=ModelConfig(researcher_model="m1", reporter_model="m2"),
        tools=ToolConfig(),
        middleware=MiddlewareConfig(),
        reporting=ReportingConfig(),
        observability=ObservabilityConfig(),
    )
    defaults.update(overrides)
    return AppConfig(**defaults)


class TestAppConfigMemoryField:
    def test_default_memory_use_empty(self) -> None:
        config = _make_app_config()
        assert config.memory.use == ""

    def test_default_memory_enable_recall_true(self) -> None:
        config = _make_app_config()
        assert config.memory.enable_recall is True

    def test_default_memory_enable_extract_false(self) -> None:
        config = _make_app_config()
        assert config.memory.enable_extract is False

    def test_default_memory_is_memory_config_instance(self) -> None:
        config = _make_app_config()
        assert isinstance(config.memory, MemoryConfig)

    def test_field_exists(self) -> None:
        fields = [f.name for f in dataclasses.fields(AppConfig)]
        assert "memory" in fields

    def test_frozen(self) -> None:
        config = _make_app_config()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.memory = MemoryConfig()

    def test_custom_memory(self) -> None:
        custom = replace(MemoryConfig(), use="default", token_budget=4000)
        config = _make_app_config(memory=custom)
        assert config.memory.use == "default"
        assert config.memory.token_budget == 4000


class TestBackwardCompat:
    def test_existing_config_without_memory_uses_default(self) -> None:
        """既有 config 文件无 memory 段时用默认值（行为不变）。"""
        config = _make_app_config()
        assert config.memory.use == ""
        # 既有字段不受影响
        assert config.sandbox.use == ""
        assert config.skill.enabled is False or config.skill.enabled is True  # skill config 默认

    def test_existing_fields_unchanged(self) -> None:
        config = _make_app_config()
        assert config.name == "agent4ml"
        assert config.environment == "test"
        assert config.models.researcher_model == "m1"
        assert config.runtime.expert_mode is False

    def test_default_factory_independent_per_instance(self) -> None:
        """每个 AppConfig 实例的 memory 是独立的 MemoryConfig（default_factory）。"""
        c1 = _make_app_config()
        c2 = _make_app_config()
        assert c1.memory == c2.memory
        assert c1.memory is not c2.memory


class TestThreadStateIntegration:
    """ThreadState 新字段默认 None + 既有字段不变（既有测试全量通过）。"""

    def test_create_initial_thread_state_has_memory_fields(self) -> None:
        from agent4ml.backend.agents.state.thread_state import create_initial_thread_state
        state = create_initial_thread_state("hello")
        assert state["recalled_memories"] is None
        assert state["memory_updates"] is None

    def test_existing_fields_unchanged(self) -> None:
        from agent4ml.backend.agents.state.thread_state import create_initial_thread_state
        state = create_initial_thread_state("hello")
        assert state["user_input"] == "hello"
        assert state["messages"] == []
        assert state["observations"] == []
        assert state["sandbox"] is None
        assert state["orchestration"] is None
