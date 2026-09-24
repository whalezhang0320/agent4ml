from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent4ml.backend.agents.sandbox.contracts import SandboxProvider
from agent4ml.backend.agents.sandbox.integration.bootstrap_sandbox import (
    _registered_providers,
    _reset_for_testing,
    register_sandbox_shutdown,
    _shutdown_all_providers,
)


@pytest.fixture(autouse=True)
def _isolate():
    """每个测试前清空 provider 列表。"""
    _reset_for_testing()
    yield
    _reset_for_testing()


class TestRegisterSandboxShutdown:
    def test_provider_added_to_list(self) -> None:
        provider = MagicMock(spec=SandboxProvider)
        register_sandbox_shutdown(provider)
        assert provider in _registered_providers

    def test_multiple_providers_accumulate(self) -> None:
        p1 = MagicMock(spec=SandboxProvider)
        p2 = MagicMock(spec=SandboxProvider)
        register_sandbox_shutdown(p1)
        register_sandbox_shutdown(p2)
        assert len(_registered_providers) == 2


class TestShutdownAllProviders:
    def test_shutdown_called(self) -> None:
        provider = MagicMock(spec=SandboxProvider)
        register_sandbox_shutdown(provider)
        _shutdown_all_providers()
        provider.shutdown.assert_called_once()

    def test_multiple_providers_all_shutdown(self) -> None:
        p1 = MagicMock(spec=SandboxProvider)
        p2 = MagicMock(spec=SandboxProvider)
        register_sandbox_shutdown(p1)
        register_sandbox_shutdown(p2)
        _shutdown_all_providers()
        p1.shutdown.assert_called_once()
        p2.shutdown.assert_called_once()

    def test_shutdown_exception_swallowed(self) -> None:
        provider = MagicMock(spec=SandboxProvider)
        provider.shutdown.side_effect = RuntimeError("shutdown failed")
        register_sandbox_shutdown(provider)
        _shutdown_all_providers()  # should not raise

    def test_clears_provider_list(self) -> None:
        provider = MagicMock(spec=SandboxProvider)
        register_sandbox_shutdown(provider)
        _shutdown_all_providers()
        assert _registered_providers == []
