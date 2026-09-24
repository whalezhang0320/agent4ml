from __future__ import annotations

import pytest

from agent4ml.backend.agents.sandbox.contracts.sandbox_backend import SandboxBackend
from agent4ml.backend.agents.sandbox.types import SandboxInfo


class _MockPlatform(SandboxBackend):
    """Mock 实现抽象方法，记录调用。"""

    def __init__(self) -> None:
        self.created: dict[str, SandboxInfo] = {}
        self.destroyed: list[str] = []

    def create(
        self,
        thread_id: str,
        sandbox_id: str,
        extra_mounts: list | None = None,
        *,
        user_id: str | None = None,
    ) -> SandboxInfo:
        if sandbox_id in self.created:
            return self.created[sandbox_id]
        info = SandboxInfo(sandbox_id=sandbox_id, sandbox_url=f"http://x:{sandbox_id}")
        self.created[sandbox_id] = info
        return info

    def destroy(self, info: SandboxInfo) -> None:
        self.destroyed.append(info.sandbox_id)
        self.created.pop(info.sandbox_id, None)

    def is_alive(self, info: SandboxInfo) -> bool:
        return info.sandbox_id in self.created

    def discover(self, sandbox_id: str) -> SandboxInfo | None:
        return self.created.get(sandbox_id)


class _UnknownAlivePlatform(_MockPlatform):
    """is_alive 返 None（未知，不误杀）。"""

    def is_alive(self, info: SandboxInfo) -> None:
        return None


class TestSandboxBackendABC:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            SandboxBackend()  # type: ignore[abstract]

    def test_list_running_default_empty(self) -> None:
        platform = _MockPlatform()
        assert platform.list_running() == []

    def test_create_returns_sandbox_info(self) -> None:
        platform = _MockPlatform()
        info = platform.create("thread-1", "sb-1")
        assert info.sandbox_id == "sb-1"
        assert info.sandbox_url == "http://x:sb-1"

    def test_create_idempotent(self) -> None:
        platform = _MockPlatform()
        info1 = platform.create("thread-1", "sb-1")
        info2 = platform.create("thread-1", "sb-1")
        assert info1 is info2

    def test_create_different_ids_distinct(self) -> None:
        platform = _MockPlatform()
        info1 = platform.create("thread-1", "sb-1")
        info2 = platform.create("thread-2", "sb-2")
        assert info1 is not info2
        assert info1.sandbox_id == "sb-1"
        assert info2.sandbox_id == "sb-2"

    def test_destroy_removes(self) -> None:
        platform = _MockPlatform()
        info = platform.create("thread-1", "sb-1")
        platform.destroy(info)
        assert "sb-1" in platform.destroyed
        assert platform.discover("sb-1") is None

    def test_is_alive_true(self) -> None:
        platform = _MockPlatform()
        info = platform.create("thread-1", "sb-1")
        assert platform.is_alive(info) is True

    def test_is_alive_false(self) -> None:
        platform = _MockPlatform()
        ghost = SandboxInfo(sandbox_id="ghost", sandbox_url="http://x")
        assert platform.is_alive(ghost) is False

    def test_is_alive_none_unknown(self) -> None:
        platform = _UnknownAlivePlatform()
        info = platform.create("thread-1", "sb-1")
        assert platform.is_alive(info) is None

    def test_discover_found(self) -> None:
        platform = _MockPlatform()
        platform.create("thread-1", "sb-1")
        found = platform.discover("sb-1")
        assert found is not None
        assert found.sandbox_id == "sb-1"

    def test_discover_not_found(self) -> None:
        platform = _MockPlatform()
        assert platform.discover("ghost") is None
