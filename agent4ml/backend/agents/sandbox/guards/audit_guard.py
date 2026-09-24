"""AuditGuard — 命令分级 + 审计日志（S8 defense-in-depth）。

组合模式：包装底层 guard（LocalSecurityGuard 或 PermissiveGuard）。
validate_command 前做三档分级，validate_path 透传底层。

- block: 破坏性命令 → raise SandboxPermissionError
- warn:  危险但非破坏 → 记审计日志放行（Docker 隔离兜底）
- pass:  其余 → 透传底层 guard

Local 模式用 AuditGuard(LocalSecurityGuard)，Docker 模式用 AuditGuard(PermissiveGuard)。
"""
from __future__ import annotations

import logging
import re
from typing import Any

from agent4ml.backend.agents.sandbox.exceptions import SandboxPermissionError

_logger = logging.getLogger(__name__)

# block 档：破坏性命令，无论 Local/Docker 都拦截
_BLOCK_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"rm\s+-rf?\s+/(?!\w)"), "rm -rf /"),
    (re.compile(r"rm\s+-rf?\s+~(?!\w)"), "rm -rf ~"),
    (re.compile(r"rm\s+-rf?\s+\$HOME"), "rm -rf $HOME"),
    (re.compile(r"rm\s+-rf?\s+/\*"), "rm -rf /*"),
    (re.compile(r"mkfs\.\w+"), "mkfs"),
    (re.compile(r"dd\s+.*\bof=/dev/"), "dd to device"),
    (re.compile(r":\s*\(\)\s*\{.*:.*\|.*:.*\}.*;"), "fork bomb"),
]

# warn 档：危险但非破坏，记日志放行（容器隔离兜底）
_WARN_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"curl\s+[^|]*\|\s*(bash|sh)\b"), "curl pipe to shell"),
    (re.compile(r"wget\s+[^|]*\|\s*(bash|sh)\b"), "wget pipe to shell"),
    (re.compile(r"\bsudo\b"), "sudo"),
    (re.compile(r"\bchmod\s+777\b"), "chmod 777"),
    (re.compile(r"\beval\b"), "eval"),
    (re.compile(r"\bexec\b(?!\s*\.py)"), "exec builtin"),
]


def _classify(command: str) -> tuple[str, str | None]:
    """返回 (level, matched_desc)。level = 'block' / 'warn' / 'pass'。"""
    for pattern, desc in _BLOCK_PATTERNS:
        if pattern.search(command):
            return ("block", desc)
    for pattern, desc in _WARN_PATTERNS:
        if pattern.search(command):
            return ("warn", desc)
    return ("pass", None)


class AuditGuard:
    """审计守卫——命令分级 + 审计日志，组合底层 guard。

    Args:
        inner: 底层 guard（LocalSecurityGuard / PermissiveGuard）
        journal: 可选 RunJournal，用于写 sandbox.command 审计事件
    """

    def __init__(self, inner: Any, journal: Any = None) -> None:
        self._inner = inner
        self._journal = journal

    def validate_path(self, path: str, *, write: bool = False) -> None:
        """路径校验透传底层 guard。"""
        self._inner.validate_path(path, write=write)

    def validate_command(self, command: str) -> None:
        """命令分级检查：block 拦截 / warn 记日志 / pass 透传。"""
        level, desc = _classify(command)
        self._audit(command, level, desc)

        if level == "block":
            raise SandboxPermissionError(
                f"dangerous command blocked: {desc}",
                path=command[:100],
                operation="validate_command",
            )
        # warn + pass → 透传底层 guard 做进一步检查
        self._inner.validate_command(command)

    def _audit(self, command: str, level: str, desc: str | None) -> None:
        """记审计日志（logger + 可选 journal）。"""
        msg = f"sandbox.command level={level}"
        if desc:
            msg += f" desc={desc}"
        msg += f" cmd={command[:100]}"

        if level == "block":
            _logger.warning(msg)
        elif level == "warn":
            _logger.warning(msg)
        else:
            _logger.debug(msg)

        if self._journal is not None:
            try:
                self._journal.append("sandbox.command", {
                    "level": level,
                    "desc": desc or "",
                    "command": command[:200],
                })
            except Exception:
                pass  # 审计日志失败不影响主流程
