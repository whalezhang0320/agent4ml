"""CodexCredentialProvider — 读 ~/.codex/auth.json 复用 Codex CLI 登录态。

设计（spec.md CredentialProvider Requirement + 参考 deer-flow credential_loader）:
- 支持 $CODEX_AUTH_PATH 覆盖路径
- 支持 legacy + nested 两种 JSON 格式：
  - legacy: {"access_token": "...", "account_id": "..."}
  - nested: {"tokens": {"access_token": "...", "account_id": "..."}}
- 凭证缺失返 None（specialist 标 disabled）
- 凭证不写 ThreadState（INV#8），只传给 specialist runtime
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from agent4ml.backend.agents.multiagent.credential_provider import Credential


@dataclass(frozen=True)
class CodexCredential(Credential):
    """Codex CLI 凭证（access_token + account_id）。

    kind="codex"（继承 Credential 基类）。
    """

    access_token: str
    account_id: str = ""


class CodexCredentialProvider:
    """读 ~/.codex/auth.json，支持 $CODEX_AUTH_PATH 覆盖。

    支持 legacy（top-level）+ nested（tokens.*）两种格式。
    凭证缺失返 None，不抛异常（specialist 标 disabled）。
    """

    def get_credential(self) -> CodexCredential | None:
        cred_path = self._resolve_path()
        data = self._load_json(cred_path)
        if data is None:
            return None

        tokens = data.get("tokens", {})
        if not isinstance(tokens, dict):
            tokens = {}

        access_token = (
            data.get("access_token")
            or data.get("token")
            or tokens.get("access_token", "")
        )
        account_id = data.get("account_id") or tokens.get("account_id", "")

        if not access_token:
            return None

        return CodexCredential(
            kind="codex",
            access_token=access_token,
            account_id=account_id,
        )

    def _resolve_path(self) -> Path:
        configured = os.getenv("CODEX_AUTH_PATH")
        if configured:
            return Path(configured).expanduser()
        return Path.home() / ".codex" / "auth.json"

    def _load_json(self, path: Path) -> dict | None:
        if not path.exists() or path.is_dir():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
