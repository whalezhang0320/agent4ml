from __future__ import annotations

import argparse

import pytest

from agent4ml.backend.agents.multiagent.mcp.specialist_mcp_server import _create_sandbox


def _make_args(
    sandbox_id: str = "abc123",
    sandbox_url: str | None = None,
    sandbox_root: str | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        sandbox_id=sandbox_id,
        sandbox_url=sandbox_url,
        sandbox_root=sandbox_root,
    )


class TestCreateSandboxDocker:
    def test_docker_url_uses_docker_runtime(self) -> None:
        from agent4ml.backend.agents.sandbox.runtimes.docker_runtime import DockerRuntime

        args = _make_args(
            sandbox_url="http://localhost:8080",
            sandbox_root="/data/aio_docker",
        )
        sandbox = _create_sandbox(args)

        assert isinstance(sandbox._runtime, DockerRuntime)

    def test_docker_url_uses_docker_path_translator(self) -> None:
        from agent4ml.backend.agents.sandbox.translators.docker_path_translator import (
            DockerPathTranslator,
        )

        args = _make_args(
            sandbox_url="http://localhost:8080",
            sandbox_root="/data/aio_docker",
        )
        sandbox = _create_sandbox(args)

        assert isinstance(sandbox._translator, DockerPathTranslator)

    def test_docker_url_uses_docker_path_guard(self) -> None:
        from agent4ml.backend.agents.sandbox.guards.docker_path_guard import (
            DockerPathGuard,
        )

        args = _make_args(
            sandbox_url="http://localhost:8080",
            sandbox_root="/data/aio_docker",
        )
        sandbox = _create_sandbox(args)

        assert isinstance(sandbox._guard._inner, DockerPathGuard)

    def test_docker_sandbox_id_preserved(self) -> None:
        args = _make_args(
            sandbox_id="myid",
            sandbox_url="http://localhost:8080",
            sandbox_root="/data",
        )
        sandbox = _create_sandbox(args)

        assert sandbox.id == "myid"


class TestCreateSandboxLocalFallback:
    def test_no_url_uses_local_runtime(self) -> None:
        from agent4ml.backend.agents.sandbox.runtimes.local_runtime import LocalRuntime

        args = _make_args()
        sandbox = _create_sandbox(args)

        assert isinstance(sandbox._runtime, LocalRuntime)

    def test_no_url_uses_local_path_translator(self) -> None:
        from agent4ml.backend.agents.sandbox.translators.local_path_translator import (
            LocalPathTranslator,
        )

        args = _make_args()
        sandbox = _create_sandbox(args)

        assert isinstance(sandbox._translator, LocalPathTranslator)

    def test_no_url_uses_local_security_guard(self) -> None:
        from agent4ml.backend.agents.sandbox.guards.local_security_guard import (
            LocalSecurityGuard,
        )

        args = _make_args()
        sandbox = _create_sandbox(args)

        assert isinstance(sandbox._guard._inner, LocalSecurityGuard)

    def test_none_url_falls_back_to_local(self) -> None:
        from agent4ml.backend.agents.sandbox.runtimes.local_runtime import LocalRuntime

        args = _make_args(sandbox_url=None)
        sandbox = _create_sandbox(args)

        assert isinstance(sandbox._runtime, LocalRuntime)
