"""凭证脱敏 — 错误信息回 LLM 前清洗 secrets。

INVARIANT:
- 正则模式：ghp_* / sk-* / Bearer * / token=* → [REDACTED]
- 仅作用于 sanitize_error（错误信息），不影响 tool result 正常内容
- check_env / scan_description no-op
"""
from __future__ import annotations

import re

_CREDENTIAL_PATTERNS = [
    re.compile(r"ghp_[A-Za-z0-9]{36}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]+"),
    re.compile(r"token=[A-Za-z0-9]+"),
]

_REDACTED = "[REDACTED]"


class CredentialSanitizer:
    """凭证脱敏 guard。sanitize_error 清洗，其余接口 no-op。"""

    def check_env(self, env: dict[str, str]) -> dict[str, str]:
        return env

    def sanitize_error(self, error_text: str) -> str:
        """正则替换凭证为 [REDACTED]。"""
        result = error_text
        for pattern in _CREDENTIAL_PATTERNS:
            result = pattern.sub(_REDACTED, result)
        return result

    def scan_description(self, tool_name: str, description: str) -> bool:
        return False
