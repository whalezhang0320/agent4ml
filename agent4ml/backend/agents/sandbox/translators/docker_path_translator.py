from __future__ import annotations

from pathlib import Path

_VIRTUAL_PREFIX = "/mnt/agent4ml/user-data"


class DockerPathTranslator:
    """Docker 路径 translator:容器内直传 + 反向映射到 host。

    translate_path 直传(容器内 /mnt/agent4ml/user-data = bind mount 物理路径)。
    reverse_translate 反向映射(供 Sandbox.get_host_path 用,artifact 提取)。

    INVARIANT:
    - translate_path 幂等(同 IdentityTranslator)
    - reverse_translate 仅接受 /mnt/agent4ml/user-data 前缀
    - sandbox_root + sandbox_id 构造时确定,不可变
    """

    def __init__(self, sandbox_root: str | Path, sandbox_id: str) -> None:
        self._host_root = str(Path(sandbox_root) / sandbox_id).replace("\\", "/")

    def translate_path(self, virtual_path: str) -> str:
        return virtual_path

    def translate_command(self, command: str) -> str:
        return command

    def mask_output(self, output: str) -> str:
        return output

    def reverse_translate(self, virtual_path: str) -> str:
        """虚拟路径 → host 物理路径(Windows 或 Linux,取决于 sandbox_root)。

        /mnt/agent4ml/user-data/foo → <sandbox_root>/<sandbox_id>/foo
        非 /mnt/agent4ml/user-data(精确匹配或带 /)前缀抛 ValueError。
        输出统一正斜杠(shutil.copy2 在 Windows 上兼容两种分隔符)。
        """
        if virtual_path != _VIRTUAL_PREFIX and not virtual_path.startswith(
            _VIRTUAL_PREFIX + "/"
        ):
            raise ValueError(f"path not under {_VIRTUAL_PREFIX}: {virtual_path}")
        relative = virtual_path[len(_VIRTUAL_PREFIX):].lstrip("/")
        return f"{self._host_root}/{relative}" if relative else self._host_root
