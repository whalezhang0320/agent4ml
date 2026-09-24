"""Stage 6 集成测试（需 Docker + agent_sandbox）。

skip if no docker or no agent_sandbox installed.
"""

from __future__ import annotations

import shutil

import pytest

agent_sandbox = pytest.importorskip("agent_sandbox")
shutil.which("docker") or pytest.skip("docker CLI not available", allow_module_level=True)

from agent4ml.backend.agents.sandbox.docker.docker_sandbox_provider import (  # noqa: E402
    DockerSandboxProvider,
)
from agent4ml.backend.agents.sandbox.runtimes.docker_runtime import DockerRuntime  # noqa: E402


@pytest.fixture
def provider() -> DockerSandboxProvider:
    return DockerSandboxProvider(
        image="all-in-one-sandbox:latest",
        base_port=19000,
        container_prefix="agent4ml-itest",
        idle_timeout=0,
    )


class TestIntegration:
    """端到端：DockerSandboxProvider acquire → Sandbox 文件操作 → release → reclaim → shutdown。"""

    def test_acquire_and_exec(self, provider: DockerSandboxProvider) -> None:
        sid = provider.acquire("itest-thread")
        try:
            sandbox = provider.get(sid)
            assert sandbox is not None
            output = sandbox.execute_command("echo hello")
            assert "hello" in output
        finally:
            provider.shutdown()

    def test_release_and_reclaim(self, provider: DockerSandboxProvider) -> None:
        sid = provider.acquire("itest-thread")
        provider.release(sid)
        # reclaim from warm pool
        sid2 = provider.acquire("itest-thread")
        assert sid == sid2
        provider.shutdown()

    def test_shutdown_destroys_all(self, provider: DockerSandboxProvider) -> None:
        sid = provider.acquire("itest-thread")
        provider.shutdown()
        # after shutdown, get returns None
        assert provider.get(sid) is None
