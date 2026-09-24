"""安全层 Protocol 基类 — 可扩展的 MCP 安全检查接口。

扩展点：新增 guard 实现 Protocol 即可，loader 按配置装载。
三个接口各 guard 按职责实现，无关接口返原值（no-op）。
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class SecurityGuard(Protocol):
    """MCP 安全检查层。可组合（多 guard 按顺序应用）。

    check_env: 过滤子进程环境变量（stdio transport 用）
    sanitize_error: 脱敏错误信息（回 LLM 前用）
    scan_description: 检测工具描述中的可疑模式（注册前用，返 True=拒绝）
    """

    def check_env(self, env: dict[str, str]) -> dict[str, str]:
        """过滤环境变量。返过滤后的 env dict。"""
        ...

    def sanitize_error(self, error_text: str) -> str:
        """脱敏错误信息。返脱敏后的字符串。"""
        ...

    def scan_description(self, tool_name: str, description: str) -> bool:
        """检测工具描述是否可疑。返 True=拒绝注册，False=放行。"""
        ...
