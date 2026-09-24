"""env 白名单过滤 — 防 stdio 子进程继承宿主 secrets。

INVARIANT:
- 安全 env 白名单：PATH/HOME/USER/LANG/LC_ALL/TERM/SHELL/TMPDIR + Windows 系统变量 + XDG_* 前缀
- config 声明的 env 合并进白名单结果（显式声明优先）
- 宿主其他变量（AWS_KEY/GITHUB_TOKEN 等）不泄露给子进程
"""
from __future__ import annotations

import os

_SAFE_ENV_KEYS = frozenset({
    # POSIX 基础
    "PATH", "HOME", "USER", "LANG", "LC_ALL", "TERM", "SHELL", "TMPDIR",
    # Windows 系统定位
    "ALLUSERSPROFILE", "APPDATA", "COMSPEC", "PROGRAMDATA",
    "PROGRAMFILES", "SYSTEMDRIVE", "SYSTEMROOT", "WINDIR",
    "USERPROFILE", "TEMP", "TMP",
})

_SAFE_ENV_PREFIXES = ("XDG_",)


class EnvFilter:
    """env 白名单 guard。check_env 过滤，其余接口 no-op。"""

    def check_env(self, env: dict[str, str]) -> dict[str, str]:
        """白名单过滤：宿主 env 仅安全键 + config 声明 env 合并。

        config 声明的 env 覆盖白名单结果（显式声明优先）。
        """
        safe: dict[str, str] = {}
        for key, value in os.environ.items():
            if key in _SAFE_ENV_KEYS or any(key.startswith(p) for p in _SAFE_ENV_PREFIXES):
                safe[key] = value
        safe.update(env)
        return safe

    def sanitize_error(self, error_text: str) -> str:
        return error_text

    def scan_description(self, tool_name: str, description: str) -> bool:
        return False
