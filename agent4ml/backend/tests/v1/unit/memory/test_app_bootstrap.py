from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("langchain_core")

from agent4ml.backend.agents.config.schema import AppConfig
from agent4ml.backend.agents.memory.bootstrap import reset_memory_provider, set_memory_provider
from agent4ml.backend.agents.memory.config import get_memory_config, set_memory_config
from agent4ml.backend.app.bootstrap import _load_memory_provider, _resolve_relative_paths


def _make_app_config(**overrides) -> AppConfig:
    from agent4ml.backend.agents.config.schema import (
        MiddlewareConfig,
        ModelConfig,
        ObservabilityConfig,
        ReportingConfig,
        RuntimeConfig,
        ToolConfig,
    )
    defaults = dict(
        name="test",
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


@pytest.fixture(autouse=True)
def _reset_provider():
    reset_memory_provider()
    yield
    reset_memory_provider()


@pytest.fixture(autouse=True)
def _reset_config():
    original = get_memory_config()
    yield
    set_memory_config(original)


class TestLoadMemoryProvider:
    def test_config_use_empty_returns_none(self) -> None:
        """config.memory.use="" 返 None（记忆禁用）。"""
        config = _make_app_config()
        config = replace(config, memory=replace(config.memory, use=""))
        assert _load_memory_provider(config) is None

    def test_config_use_default_returns_provider(self, tmp_path: Path) -> None:
        """config.memory.use="default" 返 provider。"""
        config = _make_app_config()
        config = replace(config, memory=replace(config.memory, use="default", storage_path=str(tmp_path)))
        set_memory_config(replace(get_memory_config(), use="default", storage_path=str(tmp_path)))
        provider = _load_memory_provider(config)
        assert provider is not None

    def test_uses_get_memory_provider_singleton(self) -> None:
        """_load_memory_provider 调 get_memory_provider（全局单例）。"""
        config = _make_app_config()
        config = replace(config, memory=replace(config.memory, use="default"))
        mock = object()
        set_memory_provider(mock)
        assert _load_memory_provider(config) is mock


class TestResolveRelativePaths:
    def test_anchors_memory_storage_path_to_project_root(self) -> None:
        """相对路径 storage_path 锚定 _PROJECT_ROOT。"""
        config = _make_app_config()
        config = replace(config, memory=replace(config.memory, storage_path=".agent4ml/memory"))
        result = _resolve_relative_paths(config)
        assert not Path(result.memory.storage_path).is_absolute() is False
        # 锚定后是绝对路径
        assert Path(result.memory.storage_path).is_absolute()
        p = Path(result.memory.storage_path)
        assert p.name == "memory"
        assert p.parent.name == ".agent4ml"

    def test_absolute_storage_path_unchanged(self) -> None:
        """绝对路径 storage_path 不变。"""
        config = _make_app_config()
        abs_path = str(Path.cwd() / "custom_memory")
        config = replace(config, memory=replace(config.memory, storage_path=abs_path))
        result = _resolve_relative_paths(config)
        assert result.memory.storage_path == abs_path

    def test_context_governance_still_anchored(self) -> None:
        """既有 externalize_dir 锚定不受影响。"""
        config = _make_app_config()
        config = replace(config, memory=replace(config.memory, storage_path=".agent4ml/memory"))
        result = _resolve_relative_paths(config)
        assert Path(result.context_governance.params.get("externalize_dir", "")).is_absolute()
