"""ClaudeCredentialProvider — 读 ~/.claude/.credentials.json 复用 Claude Code CLI 登录态。

设计（spec.md CredentialProvider Requirement + 参考 deer-flow credential_loader）:
- 支持 $CLAUDE_CODE_CREDENTIALS_PATH 覆盖路径
- 支持 $CLAUDE_CODE_OAUTH_TOKEN / $ANTHROPIC_AUTH_TOKEN 直接传 token
- 解析 OAuth accessToken + refreshToken + expiresAt
- 过期检测：expiresAt 是毫秒时间戳，过期返 None
- 凭证缺失返 None（specialist 标 disabled）
- 凭证不写 ThreadState（INV#8），只传给 specialist runtime
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from agent4ml.backend.agents.multiagent.credential_provider import Credential


@dataclass(frozen=True)
class ClaudeCredential(Credential):
    """Claude Code CLI OAuth 凭证。

    kind="claude"（继承 Credential 基类）。
    expires_at 是毫秒时间戳（0 表示无过期信息，不做过期检测）。
    """

    access_token: str
    refresh_token: str = ""
    expires_at: int = 0

    @property
    def is_expired(self) -> bool:
        """过期检测：expires_at 是毫秒，留 1 分钟 buffer。"""
        if self.expires_at <= 0:
            return False
        return time.time() * 1000 > self.expires_at - 60_000


class ClaudeCredentialProvider:
    """读 ~/.claude/.credentials.json，支持 env 覆盖。

    查找顺序：
    1. $CLAUDE_CODE_OAUTH_TOKEN / $ANTHROPIC_AUTH_TOKEN（直接 token）
    2. $CLAUDE_CODE_CREDENTIALS_PATH（指定 credentials 文件）
    3. ~/.claude/.credentials.json（默认路径）

    凭证缺失或过期返 None，不抛异常（specialist 标 disabled）。
    """

    def get_credential(self) -> ClaudeCredential | None:
        direct_token = (
            os.getenv("CLAUDE_CODE_OAUTH_TOKEN")
            or os.getenv("ANTHROPIC_AUTH_TOKEN")
        )
        if direct_token and direct_token.strip():
            return ClaudeCredential(
                kind="claude",
                access_token=direct_token.strip(),
            )

        cred_path = self._resolve_path()
        data = self._load_json(cred_path)
        if data is None:
            return None

        oauth = data.get("claudeAiOauth", {})
        if not isinstance(oauth, dict):
            oauth = {}
        access_token = oauth.get("accessToken", "")
        if not access_token:
            return None

        cred = ClaudeCredential(
            kind="claude",
            access_token=access_token,
            refresh_token=oauth.get("refreshToken", ""),
            expires_at=oauth.get("expiresAt", 0),
        )

        if cred.is_expired:
            return None

        return cred

    def _resolve_path(self) -> Path:
        configured = os.getenv("CLAUDE_CODE_CREDENTIALS_PATH")
        if configured:
            return Path(configured).expanduser()
        return Path.home() / ".claude" / ".credentials.json"

    def _load_json(self, path: Path) -> dict | None:
        if not path.exists() or path.is_dir():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
