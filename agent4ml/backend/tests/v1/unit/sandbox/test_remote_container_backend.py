from __future__ import annotations

import pytest

from agent4ml.backend.agents.sandbox.docker.remote_container_backend import (
    RemoteContainerBackend,
)
from agent4ml.backend.agents.sandbox.types import SandboxInfo


@pytest.fixture
def provisioner() -> RemoteContainerBackend:
    return RemoteContainerBackend()


@pytest.fixture
def info() -> SandboxInfo:
    return SandboxInfo(sandbox_id="abc123", sandbox_url="http://k3s:30001")


class TestCreate:
    def test_raises_not_implemented(self, provisioner: RemoteContainerBackend) -> None:
        with pytest.raises(NotImplementedError, match="K8s"):
            provisioner.create("thread-1", "abc123")


class TestDestroy:
    def test_raises_not_implemented(
        self, provisioner: RemoteContainerBackend, info: SandboxInfo
    ) -> None:
        with pytest.raises(NotImplementedError, match="K8s"):
            provisioner.destroy(info)


class TestIsAlive:
    def test_returns_none(
        self, provisioner: RemoteContainerBackend, info: SandboxInfo
    ) -> None:
        assert provisioner.is_alive(info) is None

    def test_does_not_raise(
        self, provisioner: RemoteContainerBackend, info: SandboxInfo
    ) -> None:
        result = provisioner.is_alive(info)
        assert result is None


class TestDiscover:
    def test_raises_not_implemented(self, provisioner: RemoteContainerBackend) -> None:
        with pytest.raises(NotImplementedError, match="K8s"):
            provisioner.discover("abc123")


class TestListRunning:
    def test_raises_not_implemented(self, provisioner: RemoteContainerBackend) -> None:
        with pytest.raises(NotImplementedError, match="K8s"):
            provisioner.list_running()
