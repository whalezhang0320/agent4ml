"""Stage 5 integration: Docker provisioner + runtime end-to-end. Skip if no docker or no agent_sandbox installed."""

from __future__ import annotations

import shutil

import pytest

agent_sandbox = pytest.importorskip("agent_sandbox")
shutil.which("docker") or pytest.skip("docker CLI not available", allow_module_level=True)

from agent4ml.backend.agents.sandbox.docker.local_container_backend import (  # noqa: E402
    LocalContainerBackend,
)
from agent4ml.backend.agents.sandbox.docker.readiness import (  # noqa: E402
    wait_for_sandbox_ready,
)
from agent4ml.backend.agents.sandbox.runtimes.docker_runtime import DockerRuntime  # noqa: E402


@pytest.fixture
def provisioner() -> LocalContainerBackend:
    return LocalContainerBackend(image="all-in-one-sandbox:latest", base_port=18000)


def _await_ready(provisioner: LocalContainerBackend, info) -> None:
    """Poll /v1/sandbox until ready (AIO HTTP server needs startup time)."""
    if not wait_for_sandbox_ready(info.sandbox_url, timeout=60):
        provisioner.destroy(info)
        pytest.skip(f"sandbox {info.sandbox_id} not ready within 60s")


class TestIntegration:
    """End-to-end DockerRuntime SDK calls + file operations against provisioner-created container."""

    def test_create_and_exec(self, provisioner: LocalContainerBackend) -> None:
        info = provisioner.create("test-thread", "itest01")
        try:
            assert info.sandbox_url
            _await_ready(provisioner, info)
            rt = DockerRuntime(info.sandbox_url)
            try:
                output = rt.exec_command("echo hello")
                assert "hello" in output
            finally:
                rt.close()
        finally:
            provisioner.destroy(info)

    def test_write_and_read(self, provisioner: LocalContainerBackend) -> None:
        info = provisioner.create("test-thread", "itest02")
        try:
            _await_ready(provisioner, info)
            rt = DockerRuntime(info.sandbox_url)
            try:
                rt.write_file("/mnt/agent4ml/user-data/test.txt", "hello world")
                content = rt.read_file("/mnt/agent4ml/user-data/test.txt")
                assert "hello world" in content
            finally:
                rt.close()
        finally:
            provisioner.destroy(info)

    def test_is_alive_and_discover(self, provisioner: LocalContainerBackend) -> None:
        info = provisioner.create("test-thread", "itest03")
        try:
            assert provisioner.is_alive(info) is True
            discovered = provisioner.discover("itest03")
            assert discovered is not None
            assert discovered.sandbox_id == "itest03"
        finally:
            provisioner.destroy(info)
            assert provisioner.is_alive(info) is False
