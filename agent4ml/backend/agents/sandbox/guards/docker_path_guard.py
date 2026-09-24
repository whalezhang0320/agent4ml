from __future__ import annotations

import re

from agent4ml.backend.agents.sandbox.exceptions import SandboxPermissionError

_VIRTUAL_PREFIX = "/mnt/agent4ml/user-data/"
_REDIRECT_PATTERN = re.compile(r'>{1,2}\s*(/[^\s;|&]*)')


class DockerPathGuard:
    """Docker 路径白名单 Guard:写入必须落挂载区。

    validate_path(write=True):路径必须 /mnt/agent4ml/user-data/ 前缀。
    validate_command:bash 重定向目标(绝对路径)必须 /mnt/agent4ml/user-data/ 前缀。
    读不限制(容器隔离兜底)。

    INVARIANT:
    - 不解析 cd 后相对路径(复杂,容器隔离兜底)
    - 不检查 tee(罕见,LLM 通常用 > 重定向)
    - 误拒抛 SandboxPermissionError(LLM 可重试改路径)
    - 只捕获绝对路径重定向,跳过 2>&1(文件描述符)+ 相对路径
    """

    def validate_path(self, path: str, *, write: bool = False) -> None:
        if not write:
            return
        if not path.startswith(_VIRTUAL_PREFIX):
            raise SandboxPermissionError(
                f"write path must be under {_VIRTUAL_PREFIX}: {path}",
                path=path,
                operation="validate",
            )

    def validate_command(self, command: str) -> None:
        for match in _REDIRECT_PATTERN.finditer(command):
            target = match.group(1)
            if not target.startswith(_VIRTUAL_PREFIX):
                raise SandboxPermissionError(
                    f"bash redirect target must be under {_VIRTUAL_PREFIX}: {target}",
                    path=target,
                    operation="validate_command",
                )
