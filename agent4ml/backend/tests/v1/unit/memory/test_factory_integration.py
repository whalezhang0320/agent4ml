from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent4ml.backend.agents.leader.factory import _build_middlewares
from agent4ml.backend.agents.middlewares.memory_recall_middleware import MemoryMiddleware
from agent4ml.backend.agents.middlewares.sandbox_middleware import SandboxMiddleware
from agent4ml.backend.agents.middlewares.tool_call_middleware import ToolCallMiddleware


def _make_memory_config(**overrides):
    defaults = {"enable_recall": True, "enable_extract": False, "token_budget": 2000}
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestBuildMiddlewaresMemory:
    def test_memory_provider_mounts_memory_middleware(self) -> None:
        """memory_provider 非 None 时挂载 MemoryMiddleware。"""
        middlewares = _build_middlewares(
            expert_mode=False,
            memory_provider=object(),
            memory_config=_make_memory_config(),
        )
        assert any(isinstance(m, MemoryMiddleware) for m in middlewares)

    def test_no_memory_provider_no_memory_middleware(self) -> None:
        """memory_provider=None 不挂载 MemoryMiddleware（既有行为不变）。"""
        middlewares = _build_middlewares(expert_mode=False, memory_provider=None)
        assert not any(isinstance(m, MemoryMiddleware) for m in middlewares)

    def test_memory_middleware_after_sandbox_before_toolcall(self) -> None:
        """挂载顺序：Sandbox 后,ToolCall 前。"""
        middlewares = _build_middlewares(
            expert_mode=False,
            sandbox_provider=object(),  # 触发 SandboxMiddleware 挂载
            artifact_server=object(),
            memory_provider=object(),
            memory_config=_make_memory_config(),
        )
        sandbox_idx = next(
            (i for i, m in enumerate(middlewares) if isinstance(m, SandboxMiddleware)), -1
        )
        memory_idx = next(
            (i for i, m in enumerate(middlewares) if isinstance(m, MemoryMiddleware)), -1
        )
        toolcall_idx = next(
            (i for i, m in enumerate(middlewares) if isinstance(m, ToolCallMiddleware)), -1
        )
        assert sandbox_idx >= 0, "SandboxMiddleware not mounted"
        assert memory_idx >= 0, "MemoryMiddleware not mounted"
        assert toolcall_idx >= 0, "ToolCallMiddleware not mounted"
        assert sandbox_idx < memory_idx < toolcall_idx

    def test_memory_config_parameterized(self) -> None:
        """参数化 enable_recall/extract/token_budget 从 memory_config 取。"""
        config = _make_memory_config(enable_recall=False, enable_extract=True, token_budget=500)
        middlewares = _build_middlewares(
            expert_mode=False,
            memory_provider=object(),
            memory_config=config,
        )
        mw = next(m for m in middlewares if isinstance(m, MemoryMiddleware))
        assert mw._enable_recall is False
        assert mw._enable_extract is True
        assert mw._token_budget == 500

    def test_memory_config_none_uses_defaults(self) -> None:
        """memory_config=None 时用默认 True/False/2000。"""
        middlewares = _build_middlewares(
            expert_mode=False,
            memory_provider=object(),
            memory_config=None,
        )
        mw = next(m for m in middlewares if isinstance(m, MemoryMiddleware))
        assert mw._enable_recall is True
        assert mw._enable_extract is False
        assert mw._token_budget == 2000
