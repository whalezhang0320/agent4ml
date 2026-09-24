from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from agent4ml.backend.agents.sandbox.docker.executor import (
    DockerExecutor,
    LocalDockerExecutor,
    WslDockerExecutor,
)


class TestLocalDockerExecutor:
    def test_is_docker_executor(self) -> None:
        assert isinstance(LocalDockerExecutor(), DockerExecutor)

    @patch("agent4ml.backend.agents.sandbox.docker.executor.subprocess.run")
    def test_run_passes_cmd_directly(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(stdout="ok", returncode=0)
        LocalDockerExecutor().run(["docker", "ps"], capture_output=True, text=True)
        mock_run.assert_called_once_with(["docker", "ps"], capture_output=True, text=True)

    def test_translate_path_identity(self) -> None:
        ex = LocalDockerExecutor()
        assert ex.translate_path("/home/user/data") == "/home/user/data"

    def test_translate_path_windows_passed_through(self) -> None:
        ex = LocalDockerExecutor()
        assert ex.translate_path("D:\\foo\\bar") == "D:\\foo\\bar"


class TestWslDockerExecutor:
    def test_is_docker_executor(self) -> None:
        assert isinstance(WslDockerExecutor(), DockerExecutor)

    @patch("agent4ml.backend.agents.sandbox.docker.executor.subprocess.run")
    def test_run_with_user_prefix(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        WslDockerExecutor(distro="Ubuntu", user="root").run(
            ["docker", "ps"], capture_output=True, text=True,
        )
        mock_run.assert_called_once_with(
            ["wsl", "-d", "Ubuntu", "--user", "root", "--", "docker", "ps"],
            capture_output=True, text=True,
        )

    @patch("agent4ml.backend.agents.sandbox.docker.executor.subprocess.run")
    def test_run_without_user(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        WslDockerExecutor(distro="Ubuntu").run(["docker", "ps"])
        args = mock_run.call_args[0][0]
        assert args == ["wsl", "-d", "Ubuntu", "--", "docker", "ps"]
        assert "--user" not in args

    def test_translate_path_drive_letter(self) -> None:
        ex = WslDockerExecutor()
        assert ex.translate_path("D:\\ProjectFile\\Agent4ML\\.agent4ml\\sandbox\\sb1") == (
            "/mnt/d/ProjectFile/Agent4ML/.agent4ml/sandbox/sb1"
        )

    def test_translate_path_lowercase_drive(self) -> None:
        ex = WslDockerExecutor()
        result = ex.translate_path("C:\\Users\\test")
        assert result == "/mnt/c/Users/test"

    def test_translate_path_no_drive_letter_passthrough(self) -> None:
        ex = WslDockerExecutor()
        assert ex.translate_path("/already/posix/path") == "/already/posix/path"

    def test_translate_path_forward_slashes_in_windows_path(self) -> None:
        ex = WslDockerExecutor()
        assert ex.translate_path("D:/foo/bar") == "/mnt/d/foo/bar"

    def test_default_distro_is_ubuntu(self) -> None:
        ex = WslDockerExecutor()
        assert ex._prefix == ["wsl", "-d", "Ubuntu"]
