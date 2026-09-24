"""MCP 安全层 — Protocol 基类 + 3 内置 guard。

INVARIANT:
- SecurityGuard Protocol 定义 3 接口：check_env / sanitize_error / scan_description
- EnvFilter: env 白名单过滤（防宿主 secrets 泄露给 stdio 子进程）
- CredentialSanitizer: 凭证脱敏（错误信息回 LLM 前清洗）
- DescriptionScanner: 描述扫描（注册时检测 prompt injection）
- loader 按顺序应用 guards
"""
from agent4ml.backend.agents.mcp.guards.base import SecurityGuard
from agent4ml.backend.agents.mcp.guards.credential_sanitizer import CredentialSanitizer
from agent4ml.backend.agents.mcp.guards.description_scanner import DescriptionScanner
from agent4ml.backend.agents.mcp.guards.env_filter import EnvFilter

__all__ = [
    "SecurityGuard",
    "EnvFilter",
    "CredentialSanitizer",
    "DescriptionScanner",
]
