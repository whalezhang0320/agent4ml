from __future__ import annotations

import pytest

from agent4ml.backend.agents.sandbox.contracts import SecurityGuard
from agent4ml.backend.agents.sandbox.exceptions import SandboxPermissionError
from agent4ml.backend.agents.sandbox.guards.docker_path_guard import DockerPathGuard


class TestDockerPathGuard:
    def test_is_security_guard(self) -> None:
        guard = DockerPathGuard()
        assert isinstance(guard, SecurityGuard)

    def test_validate_path_write_mount_area_passes(self) -> None:
        guard = DockerPathGuard()
        guard.validate_path("/mnt/agent4ml/user-data/foo", write=True)

    def test_validate_path_write_non_mount_raises(self) -> None:
        guard = DockerPathGuard()
        with pytest.raises(SandboxPermissionError, match="write path must be under"):
            guard.validate_path("/tmp/foo", write=True)

    def test_validate_path_write_root_path_raises(self) -> None:
        guard = DockerPathGuard()
        with pytest.raises(SandboxPermissionError):
            guard.validate_path("/mnt/agent4ml/user-data", write=True)

    def test_validate_path_read_not_restricted(self) -> None:
        guard = DockerPathGuard()
        guard.validate_path("/tmp/foo", write=False)
        guard.validate_path("/etc/passwd", write=False)
        guard.validate_path("/any/path", write=False)

    def test_validate_command_redirect_to_mount_passes(self) -> None:
        guard = DockerPathGuard()
        guard.validate_command("echo foo > /mnt/agent4ml/user-data/bar")

    def test_validate_command_append_to_mount_passes(self) -> None:
        guard = DockerPathGuard()
        guard.validate_command("echo foo >> /mnt/agent4ml/user-data/bar")

    def test_validate_command_redirect_to_tmp_raises(self) -> None:
        guard = DockerPathGuard()
        with pytest.raises(SandboxPermissionError, match="bash redirect target"):
            guard.validate_command("echo foo > /tmp/bar")

    def test_validate_command_append_to_tmp_raises(self) -> None:
        guard = DockerPathGuard()
        with pytest.raises(SandboxPermissionError):
            guard.validate_command("echo foo >> /tmp/bar")

    def test_validate_command_stderr_redirect_not_rejected(self) -> None:
        guard = DockerPathGuard()
        guard.validate_command("cmd 2>&1")

    def test_validate_command_no_redirect_passes(self) -> None:
        guard = DockerPathGuard()
        guard.validate_command("ls -la /mnt/agent4ml/user-data")
        guard.validate_command("cat /mnt/agent4ml/user-data/foo.txt")
        guard.validate_command("pip install package")

    def test_validate_command_relative_redirect_not_rejected(self) -> None:
        guard = DockerPathGuard()
        guard.validate_command("echo foo > bar.txt")

    def test_validate_command_redirect_with_pipe(self) -> None:
        guard = DockerPathGuard()
        with pytest.raises(SandboxPermissionError):
            guard.validate_command("echo foo | tee /tmp/out > /tmp/bar")

    def test_validate_command_multiple_redirects(self) -> None:
        guard = DockerPathGuard()
        with pytest.raises(SandboxPermissionError):
            guard.validate_command(
                "echo a > /mnt/agent4ml/user-data/a; echo b > /tmp/b"
            )

    def test_validate_command_complex_command_mount_ok(self) -> None:
        guard = DockerPathGuard()
        guard.validate_command(
            "find /mnt/agent4ml/user-data -name '*.py' > /mnt/agent4ml/user-data/out.txt"
        )

    def test_validate_path_write_deep_nested_passes(self) -> None:
        guard = DockerPathGuard()
        guard.validate_path(
            "/mnt/agent4ml/user-data/workspace/project/src/main.py", write=True
        )
