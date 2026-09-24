from __future__ import annotations

from unittest.mock import MagicMock

from agent4ml.backend.agents.multiagent.runtimes import append_sandbox_url_args


class TestAppendSandboxUrlArgs:
    def test_none_provider_no_append(self) -> None:
        cmd: list[str] = []
        append_sandbox_url_args(cmd, None, "abc")
        assert cmd == []

    def test_docker_provider_appends_url_and_root(self) -> None:
        provider = MagicMock()
        info = MagicMock()
        info.sandbox_url = "http://localhost:8080"
        provider.get_sandbox_info.return_value = info
        provider._sandbox_root = "/data/aio_docker"

        cmd: list[str] = []
        append_sandbox_url_args(cmd, provider, "abc")

        assert "--sandbox-url" in cmd
        assert "http://localhost:8080" in cmd
        assert "--sandbox-root" in cmd
        assert "/data/aio_docker" in cmd

    def test_local_provider_returns_none_no_append(self) -> None:
        provider = MagicMock()
        provider.get_sandbox_info.return_value = None

        cmd: list[str] = []
        append_sandbox_url_args(cmd, provider, "abc")

        assert cmd == []

    def test_empty_sandbox_url_no_append(self) -> None:
        provider = MagicMock()
        info = MagicMock()
        info.sandbox_url = ""
        provider.get_sandbox_info.return_value = info

        cmd: list[str] = []
        append_sandbox_url_args(cmd, provider, "abc")

        assert cmd == []

    def test_no_sandbox_root_attr_only_url(self) -> None:
        provider = MagicMock()
        info = MagicMock()
        info.sandbox_url = "http://localhost:8080"
        provider.get_sandbox_info.return_value = info
        del provider._sandbox_root

        cmd: list[str] = []
        append_sandbox_url_args(cmd, provider, "abc")

        assert "--sandbox-url" in cmd
        assert "http://localhost:8080" in cmd
        assert "--sandbox-root" not in cmd
