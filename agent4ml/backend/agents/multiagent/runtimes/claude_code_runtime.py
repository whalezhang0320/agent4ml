"""ClaudeCodeRuntime — CLI 直接调实现（claude --print 子进程 + stdout 解析）。

设计（spec.md ClaudeCodeRuntime Requirement + design.md §2）:
- sync only MVP（INV#5）
- 不走 ACP（无 Claude Code ACP adapter）
- 配置 claude mcp add 用 Agent4ML MCP tool（SpecialistMcpServer）
- 超时 kill + crash SpecialistCrashError
- specialist 黑盒：claude CLI 自带 model + 自管 ReAct loop（INV#1/INV#2）
- Bug C 修复（设计文档 46 §4.3）：_build_env 透传 auth-related env vars
"""
from __future__ import annotations

import os
import subprocess
import time

from agent4ml.backend.agents.multiagent.exceptions import (
    SpecialistCrashError,
    SpecialistStartupError,
    SpecialistTimeoutError,
)
from agent4ml.backend.agents.multiagent.runtimes import append_sandbox_url_args
from agent4ml.backend.agents.multiagent.types import (
    SpecialistRawResult,
    SpecialistRequest,
)


class ClaudeCodeRuntime:
    """CLI 直接调 runtime（claude --print + stdout 解析）。

    sync only MVP（INV#5）。每次 invoke 启动新进程。
    """

    def __init__(
        self, command: str = "claude", sandbox_provider=None
    ) -> None:
        self._command = command
        self._sandbox_provider = sandbox_provider

    def invoke(self, request: SpecialistRequest) -> SpecialistRawResult:
        start = time.time()
        cmd = self._build_command(request)
        env = self._build_env()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=request.timeout_seconds,
                env=env,
            )
        except subprocess.TimeoutExpired:
            raise SpecialistTimeoutError(
                timeout_seconds=request.timeout_seconds,
            )
        except FileNotFoundError:
            raise SpecialistStartupError(
                f"claude command not found: {self._command}"
            )

        if result.returncode != 0:
            raise SpecialistCrashError(
                f"claude exited with code {result.returncode}",
                exit_code=result.returncode,
            )

        return SpecialistRawResult(
            raw_output=result.stdout,
            duration_seconds=time.time() - start,
        )

    def _build_command(self, request: SpecialistRequest) -> list[str]:
        """Build claude --print command with goal as argument."""
        return [self._command, "--print", request.goal]

    def _build_env(self) -> dict[str, str] | None:
        """Pass auth-related env vars to claude subprocess.

        Bug C 修复（设计文档 46 §4.3）：
        透传 CLAUDE_CODE_OAUTH_TOKEN + ANTHROPIC_AUTH_TOKEN + ANTHROPIC_API_KEY
        + CLAUDE_CODE_CREDENTIALS_PATH 给 claude CLI 子进程。

        返 merge 后的 env（父进程 env + auth vars 覆盖），保证子进程 PATH/HOME 等基础 env 可用。
        无 auth vars 时返 None（subprocess.run 用 None，子进程继承父 env）。
        """
        auth_env: dict[str, str] = {}
        for var in (
            "CLAUDE_CODE_OAUTH_TOKEN",
            "ANTHROPIC_AUTH_TOKEN",
            "ANTHROPIC_API_KEY",
            "CLAUDE_CODE_CREDENTIALS_PATH",
        ):
            val = os.getenv(var)
            if val:
                auth_env[var] = val
        if not auth_env:
            return None
        # merge：父进程 env + auth vars 覆盖（保证 PATH/HOME 等基础 env 可用）
        return {**os.environ, **auth_env}

    def _build_mcp_add_command(self, sandbox_id: str) -> list[str]:
        """Build `claude mcp add` command for SpecialistMcpServer."""
        cmd = [
            self._command,
            "mcp",
            "add",
            "agent4ml_sandbox",
            "--",
            "python",
            "-m",
            "agent4ml.backend.agents.multiagent.mcp.specialist_mcp_server",
            "--sandbox-id",
            sandbox_id,
        ]
        append_sandbox_url_args(cmd, self._sandbox_provider, sandbox_id)
        return cmd

    def configure_mcp(self, sandbox_id: str) -> None:
        """Run `claude mcp add` to register SpecialistMcpServer.

        Called by specialist before invoke (or by bootstrap).
        """
        cmd = self._build_mcp_add_command(sandbox_id)
        subprocess.run(cmd, capture_output=True, timeout=30)
