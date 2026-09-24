"""sandbox_id 格式校验工具（S6 defense-in-depth）。

sandbox_id 来自 _deterministic_sandbox_id（sha256[:8]），当前安全。
但反序列化路径（SandboxInfo.from_dict 从文件恢复）可能引入不可信值。
路径拼接 Path(root) / sandbox_id 若含 ../ 则 path traversal。
"""
from __future__ import annotations

import re

_SANDBOX_ID_RE = re.compile(r"^[a-f0-9]{8}$")


def validate_sandbox_id(sandbox_id: str) -> None:
    """校验 sandbox_id 格式：必须是 8 位小写十六进制（sha256[:8]）。

    Raises:
        ValueError: sandbox_id 不匹配 ^[a-f0-9]{8}$
    """
    if not isinstance(sandbox_id, str) or not _SANDBOX_ID_RE.match(sandbox_id):
        raise ValueError(f"invalid sandbox_id: {sandbox_id!r}")
