from __future__ import annotations

import re
import shlex
from pathlib import PurePosixPath

from agent4ml.backend.agents.sandbox.exceptions import SandboxPermissionError
from agent4ml.backend.agents.sandbox.types import PathMapping

_SYSTEM_PATH_PREFIXES = ("/bin/", "/usr/", "/lib/")

# 危险模式黑名单（S4 defense-in-depth）。
# 每项 (regex, description)。regex 用 word-boundary 降低误报。
# 非唯一防线——真正隔离靠 allow_host_bash gate（S2）。
_DANGEROUS_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # 破坏性命令
    (re.compile(r"rm\s+-rf?\s+/(?!\w)"), "rm -rf /"),
    (re.compile(r"rm\s+-rf?\s+~(?!\w)"), "rm -rf ~"),
    (re.compile(r"rm\s+-rf?\s+\$HOME"), "rm -rf $HOME"),
    (re.compile(r"rm\s+-rf?\s+/\*"), "rm -rf /*"),
    (re.compile(r"mkfs\.\w+"), "mkfs filesystem format"),
    (re.compile(r"dd\s+.*\bof=/dev/"), "dd to device"),
    (re.compile(r":\s*\(\)\s*\{.*:.*\|.*:.*\}.*;"), "fork bomb"),
    # 远程代码执行
    (re.compile(r"curl\s+[^|]*\|\s*(bash|sh)\b"), "curl pipe to shell"),
    (re.compile(r"wget\s+[^|]*\|\s*(bash|sh)\b"), "wget pipe to shell"),
    # 危险内置命令
    (re.compile(r"\beval\b"), "eval builtin"),
    (re.compile(r"\bexec\b(?!\s*\.py)"), "exec builtin"),
    (re.compile(r"\bsource\b"), "source builtin"),
]


class LocalSecurityGuard:
    """LocalSecurityGuard — 严格白名单 + 路径穿越拒绝 + bash 命令扫描。

    方案 C 三组件之一。Local 严格白名单；Docker/E2B 用 PermissiveGuard。

    INVARIANT:
    - validate_path 拒 .. 段（路径穿越）
    - validate_path 白名单前缀检查（/mnt/agent4ml/user-data 读写，/mnt/skills 只读）
    - validate_command shlex 失败 fail-closed（raise 非 return，S4 修复）
    - validate_command 危险模式黑名单拦截（S4 defense-in-depth）
    - validate_command 扫描 bash 命令绝对路径，仅允许白名单前缀 + 系统路径
    - 系统路径放行 /bin/ /usr/ /lib/，不放行 /tmp/（Grill #6）
    - 只做 validate，不做 mask_output（Grill #4）
    - 验证失败抛 SandboxPermissionError
    """

    def __init__(self, path_mappings: list[PathMapping]) -> None:
        self._mappings = path_mappings

    def _reject_path_traversal(self, path: str) -> None:
        parts = PurePosixPath(path).parts
        if ".." in parts:
            raise SandboxPermissionError(
                f"path traversal detected: {path}", path=path, operation="validate"
            )

    def _find_mapping(self, path: str) -> PathMapping | None:
        """找 path 匹配的 PathMapping（最长前缀优先）。"""
        for mapping in sorted(
            self._mappings, key=lambda m: len(m.container_path), reverse=True
        ):
            container = mapping.container_path.rstrip("/")
            if path == container or path.startswith(container + "/"):
                return mapping
        return None

    def validate_path(self, path: str, *, write: bool = False) -> None:
        self._reject_path_traversal(path)
        mapping = self._find_mapping(path)
        if mapping is None:
            raise SandboxPermissionError(
                f"path not in whitelist: {path}", path=path, operation="validate"
            )
        if write and mapping.read_only:
            raise SandboxPermissionError(
                f"write to read-only path: {path}", path=path, operation="write"
            )

    def validate_command(self, command: str) -> None:
        """扫描 bash 命令：危险模式黑名单 + 绝对路径白名单。

        shlex 失败 → fail-closed（raise SandboxPermissionError），不放过畸形引号攻击。
        """
        # 危险模式黑名单（先于 shlex，直接 pattern match 整条命令）
        for pattern, desc in _DANGEROUS_PATTERNS:
            if pattern.search(command):
                raise SandboxPermissionError(
                    f"dangerous command blocked: {desc}",
                    path=command[:100],
                    operation="validate_command",
                )

        # shlex 解析 → 绝对路径白名单检查
        try:
            tokens = shlex.split(command)
        except ValueError:
            # fail-closed：shlex 解析失败（畸形引号）→ 拒绝，不放过
            raise SandboxPermissionError(
                f"command has unparseable quoting: {command[:100]}",
                path=command[:100],
                operation="validate_command",
            )
        for token in tokens:
            if token.startswith("/"):
                if any(token.startswith(p) for p in _SYSTEM_PATH_PREFIXES):
                    continue
                mapping = self._find_mapping(token)
                if mapping is None:
                    raise SandboxPermissionError(
                        f"command references path not in whitelist: {token}",
                        path=token,
                        operation="validate_command",
                    )
