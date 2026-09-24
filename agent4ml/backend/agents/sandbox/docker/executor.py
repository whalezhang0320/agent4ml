"""Docker command executor layer — abstracts where docker CLI runs.

Decouples container management (provisioner) from command execution environment
(local subprocess, WSL bridge, future SSH/remote).
"""

from __future__ import annotations

import subprocess
from typing import Protocol, runtime_checkable


@runtime_checkable
class DockerExecutor(Protocol):
    """Abstraction over docker CLI invocation + host-path translation.

    run(): execute a docker command (cmd is the bare docker args, executor adds prefix).
    translate_path(): convert a host path (project process view) to daemon-view path.
    """

    def run(self, cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
        """Execute docker command, kwargs match subprocess.run signature."""
        ...

    def translate_path(self, host_path: str) -> str:
        """Translate host path to docker-daemon-visible path."""
        ...


class LocalDockerExecutor:
    """Default executor: direct subprocess.run, same-OS daemon."""

    def run(self, cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, **kwargs)

    def translate_path(self, host_path: str) -> str:
        return host_path


class WslDockerExecutor:
    """Executor for Windows project + docker daemon in WSL2.

    run(): prefixes `wsl -d <distro> [--user <user>] --` to docker command.
    translate_path(): converts Windows drive path D:\\foo\\bar → /mnt/d/foo/bar.
    """

    def __init__(self, distro: str = "Ubuntu", user: str | None = None) -> None:
        self._prefix = ["wsl", "-d", distro]
        if user:
            self._prefix += ["--user", user]

    def run(self, cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
        return subprocess.run(self._prefix + ["--"] + cmd, **kwargs)

    def translate_path(self, host_path: str) -> str:
        p = str(host_path).replace("\\", "/")
        if len(p) >= 2 and p[1] == ":":
            drive = p[0].lower()
            return f"/mnt/{drive}{p[2:]}"
        return p
