"""描述扫描 — 注册前检测工具描述中的 prompt injection。

INVARIANT:
- 检测模式：ignore previous instructions / system prompt / you are now / forget everything
- 检测到可疑模式返 True（拒绝注册），False 放行
- check_env / sanitize_error no-op
"""
from __future__ import annotations

import re

_SUSPICIOUS_PATTERNS = [
    re.compile(r"(?i)ignore\s+(previous|above|prior)\s+instructions"),
    re.compile(r"(?i)system\s+prompt"),
    re.compile(r"(?i)you\s+are\s+now"),
    re.compile(r"(?i)forget\s+(everything|prior|all)"),
    re.compile(r"(?i)reveal\s+(your|the)\s+(instructions?|prompt)"),
]


class DescriptionScanner:
    """描述扫描 guard。scan_description 检测，其余接口 no-op。"""

    def check_env(self, env: dict[str, str]) -> dict[str, str]:
        return env

    def sanitize_error(self, error_text: str) -> str:
        return error_text

    def scan_description(self, tool_name: str, description: str) -> bool:
        """检测到可疑模式返 True（拒绝注册），False 放行。"""
        for pattern in _SUSPICIOUS_PATTERNS:
            if pattern.search(description):
                return True
        return False
