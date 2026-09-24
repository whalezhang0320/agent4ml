from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class SandboxMountConfig:
    """自定义 bind mount 配置。"""

    host_path: str
    container_path: str
    read_only: bool = False


@dataclass(frozen=True)
class SandboxConfig:
    """Sandbox 配置（startup-only：变更需重启）。

    未配置（use 为空）时 sandbox 系统不启用。
    Docker 专属字段（image/port/container_prefix/idle_timeout/replicas/provisioner_url）预留（Stage 5）。
    """

    use: str = ""
    allow_host_bash: bool = True
    storage_path: str = ""
    mounts: list[SandboxMountConfig] = field(default_factory=list)
    environment: dict[str, str] = field(default_factory=dict)
    image: str = ""
    port: int = 8080
    container_prefix: str = "agent4ml-sandbox"
    idle_timeout: int = 600
    replicas: int = 3
    provisioner_url: str = ""
    executor: Literal["local", "wsl"] = "local"
    wsl_distro: str | None = None
    wsl_user: str | None = None


STARTUP_ONLY_FIELDS: set[str] = {"sandbox"}
