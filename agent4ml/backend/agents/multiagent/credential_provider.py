"""CredentialProvider Protocol — 凭证发现契约。

设计（spec.md CredentialProvider Requirement + design.md §2）:
- 凭证不进 LLM 主态：CredentialProvider 返回 token 只传给 specialist runtime，
  不写 ThreadState（INV#8）
- 复用 CLI 登录态：CodexCredentialProvider 读 ~/.codex/auth.json，
  ClaudeCredentialProvider 读 ~/.claude/.credentials.json
- Agent4ML 不管理凭证：只发现，不刷新 / 不存储
- 凭证缺失返 None：specialist 标 disabled，tool 不注册

Credential 基类：Batch 4 定义 CodexCredential / ClaudeCredential 子类。
"""
from __future__ import annotations

from dataclasses import dataclass

from typing import Protocol


@dataclass(frozen=True)
class Credential:
    """specialist 凭证基类（INV#8：不写 ThreadState，只传给 specialist runtime）。

    kind 标识 specialist 类型（"codex" / "claude"）。
    Batch 4 定义 CodexCredential / ClaudeCredential 子类扩展具体字段。
    """

    kind: str


class CredentialProvider(Protocol):
    """specialist 凭证发现契约（复用 CLI 登录态，不管理凭证）。

    实现示例：CodexCredentialProvider / ClaudeCredentialProvider（Batch 4）。
    """

    def get_credential(self) -> Credential | None:
        """发现 specialist 凭证。

        返 None 表示凭证缺失（specialist 标 disabled，tool 不注册）。
        返回的 Credential 只传给 specialist runtime，不写 ThreadState（INV#8）。
        """
        ...
