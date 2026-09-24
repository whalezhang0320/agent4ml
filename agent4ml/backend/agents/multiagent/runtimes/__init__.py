"""Multi-Agent Runtime 层 — specialist 裸执行实现。

CodexRuntime（ACP）/ ClaudeCodeRuntime（CLI）/ PiRuntime（RPC mode）/ SubagentRuntime（进程内）。
sync only MVP（INV#5），每次 invoke 启动 + 完成关闭（不做 pool）。
"""
from __future__ import annotations

from typing import Any


def append_sandbox_url_args(
    cmd_args: list[str],
    sandbox_provider: Any | None,
    sandbox_id: str,
) -> None:
    """向 MCP command args 追加 --sandbox-url + --sandbox-root（若 provider 有 SandboxInfo）。

    块 D3：specialist runtime 透传 sandbox_url，让 specialist_mcp_server 连 lead Docker 容器。
    provider 为 None 或 get_sandbox_info 返 None（Local 模式）时不追加（走 fallback LocalRuntime）。
    """
    if sandbox_provider is None:
        return
    info = sandbox_provider.get_sandbox_info(sandbox_id)
    if info is None or not info.sandbox_url:
        return
    cmd_args.extend(["--sandbox-url", info.sandbox_url])
    sandbox_root = getattr(sandbox_provider, "_sandbox_root", None)
    if sandbox_root is not None:
        cmd_args.extend(["--sandbox-root", str(sandbox_root)])
