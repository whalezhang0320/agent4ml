from __future__ import annotations


class PermissiveGuard:
    """Permissive 守卫。Docker/E2B 用，容器内已隔离，全放行。

    容器边界本身是安全边界（namespace + cgroup），
    路径白名单验证由容器负责，Guard 层 no-op。
    """

    def validate_path(self, path: str, *, write: bool = False) -> None:
        pass

    def validate_command(self, command: str) -> None:
        pass
