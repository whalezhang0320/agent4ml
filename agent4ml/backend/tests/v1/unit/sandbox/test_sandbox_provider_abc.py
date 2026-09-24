from __future__ import annotations

import asyncio

import pytest

from agent4ml.backend.agents.sandbox.contracts.sandbox_provider import SandboxProvider


class _MockProvider(SandboxProvider):
    """Mock 实现 abstract 方法，记录调用。"""

    def __init__(self) -> None:
        self.acquired: list[tuple[str | None, str | None]] = []
        self.released: list[str] = []

    def acquire(
        self, thread_id: str | None = None, *, user_id: str | None = None
    ) -> str:
        self.acquired.append((thread_id, user_id))
        return "sb-mock"

    def get(self, sandbox_id: str) -> None:
        return None

    def release(self, sandbox_id: str) -> None:
        self.released.append(sandbox_id)


class TestSandboxProviderABC:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            SandboxProvider()  # type: ignore[abstract]

    def test_capability_flags_default(self) -> None:
        provider = _MockProvider()
        assert provider.uses_thread_data_mounts is False
        assert provider.needs_upload_permission_adjustment is True

    def test_acquire_returns_id(self) -> None:
        provider = _MockProvider()
        result = provider.acquire("thread-1", user_id="user-1")
        assert result == "sb-mock"
        assert provider.acquired == [("thread-1", "user-1")]

    def test_acquire_no_args(self) -> None:
        provider = _MockProvider()
        result = provider.acquire()
        assert result == "sb-mock"
        assert provider.acquired == [(None, None)]

    def test_acquire_async_delegates_to_thread(self) -> None:
        provider = _MockProvider()
        result = asyncio.run(provider.acquire_async("thread-1", user_id="user-1"))
        assert result == "sb-mock"
        assert provider.acquired == [("thread-1", "user-1")]

    def test_acquire_async_no_args(self) -> None:
        provider = _MockProvider()
        result = asyncio.run(provider.acquire_async())
        assert result == "sb-mock"
        assert provider.acquired == [(None, None)]

    def test_get_returns_none(self) -> None:
        provider = _MockProvider()
        assert provider.get("unknown") is None

    def test_release_records(self) -> None:
        provider = _MockProvider()
        provider.release("sb-1")
        assert provider.released == ["sb-1"]

    def test_reset_default_noop(self) -> None:
        provider = _MockProvider()
        provider.reset()

    def test_shutdown_default_noop(self) -> None:
        provider = _MockProvider()
        provider.shutdown()
