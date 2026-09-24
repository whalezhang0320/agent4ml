"""PiRuntime — RPC mode 协议实现（pi --mode rpc 子进程 + stdio JSON-RPC）。

设计（spec.md PiRuntime Requirement + design_docs/46 §10.3）:
- sync only MVP（INV#5，同 CodexRuntime/ClaudeCodeRuntime）
- 每次 invoke 启动新 pi 子进程 + 完成关闭（不做 pool）
- 传 goal + context_summary + success_criteria（决策 5：MVP 不定制 system prompt，
  靠 _build_prompt 在 user message 塞 context + 三段输出格式要求）
- 超时 kill + crash SpecialistCrashError
- pi 黑盒：pi CLI 自带 model + 自管 ReAct loop（INV#1/INV#2）
- 决策 1：强制走 Agent4ML SpecialistMcpServer（--no-builtin-tools + -e agent4ml-sandbox-bridge）
- 决策 3：凭证 env 透传（国内 provider 优先）
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Any

from agent4ml.backend.agents.multiagent.exceptions import (
    SpecialistCrashError,
    SpecialistError,
    SpecialistStartupError,
    SpecialistTimeoutError,
)
from agent4ml.backend.agents.multiagent.runtimes import append_sandbox_url_args
from agent4ml.backend.agents.multiagent.types import (
    SpecialistRawResult,
    SpecialistRequest,
    TokenUsage,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PiRuntimeConfig:
    """Pi runtime 配置（决策 5：不定制 system prompt，靠 _build_prompt）。

    provider/model/thinking_level 留空时用 pi 默认（pi 自己选）。
    """

    command: str = "pi"
    mode_args: tuple[str, ...] = ("--mode", "rpc", "--no-session")
    provider: str | None = None
    model: str | None = None
    thinking_level: str = "medium"
    extra_args: tuple[str, ...] = ()


class PiRuntime:
    """Pi coding agent runtime via RPC mode (stdio JSON-RPC).

    sync only MVP（同 CodexRuntime/ClaudeCodeRuntime）。
    每次 invoke 启动新 pi 子进程 + 完成关闭。

    决策 1（设计文档 46 §10.5）：强制走 Agent4ML SpecialistMcpServer——
    --no-builtin-tools 禁用 pi 自带 read/write/edit/bash，
    -e agent4ml-sandbox-bridge 加载 Agent4ML sandbox bridge extension，
    所有操作走 Agent4ML MCP 8 接口（经过 PathTranslator + SecurityGuard）。
    """

    def __init__(self, config: PiRuntimeConfig | None = None, sandbox_provider=None) -> None:
        self._config = config or PiRuntimeConfig()
        self._sandbox_provider = sandbox_provider

    def invoke(self, request: SpecialistRequest) -> SpecialistRawResult:
        start = time.time()
        try:
            raw_output, usage = self._run_rpc_session(request)
        except subprocess.TimeoutExpired:
            raise SpecialistTimeoutError(
                timeout_seconds=request.timeout_seconds,
            )
        except FileNotFoundError:
            raise SpecialistStartupError(
                f"pi command not found: {self._config.command}. "
                "Install: npm install -g @earendil-works/pi-coding-agent"
            )
        except SpecialistError:
            raise
        except Exception as e:
            raise SpecialistCrashError(str(e))

        return SpecialistRawResult(
            raw_output=raw_output,
            usage=usage,
            duration_seconds=time.time() - start,
        )

    def _run_rpc_session(
        self, request: SpecialistRequest
    ) -> tuple[str, TokenUsage | None]:
        """启动 pi --mode rpc 子进程，发 prompt，收 events，返 final text + usage。"""
        cmd = self._build_command(request)
        env = self._build_env(request)

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        try:
            prompt = self._build_prompt(request)
            cmd_json = json.dumps(
                {"type": "prompt", "message": prompt, "id": "req-1"}
            )
            proc.stdin.write(cmd_json + "\n")
            proc.stdin.flush()

            chunks: list[str] = []
            usage_data: dict[str, Any] = {}
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                etype = event.get("type")
                if etype == "message_update":
                    ame = event.get("assistantMessageEvent", {})
                    if ame.get("type") == "text_delta":
                        chunks.append(ame["delta"])

                elif etype == "agent_end":
                    # 从最后一条 assistant message 提取 usage
                    msgs = event.get("messages", [])
                    for msg in reversed(msgs):
                        if msg.get("role") == "assistant":
                            usage_data = msg.get("usage", {}) or {}
                            break
                    break

                elif etype == "extension_error":
                    raise SpecialistCrashError(
                        f"pi extension error: {event.get('error', 'unknown')}"
                    )

            raw_output = "".join(chunks)
            usage = self._extract_usage(usage_data)
            return raw_output, usage

        finally:
            proc.stdin.close()
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    def _build_command(self, request: SpecialistRequest) -> list[str]:
        """组装 pi CLI 命令（决策 1：--no-builtin-tools + -e agent4ml-sandbox-bridge）。"""
        cmd = [self._config.command, *self._config.mode_args]
        if self._config.provider:
            cmd.extend(["--provider", self._config.provider])
        if self._config.model:
            cmd.extend(["--model", self._config.model])
        if self._config.thinking_level:
            cmd.extend(["--thinking", self._config.thinking_level])
        # 决策 1：禁用 pi 自带工具，加载 Agent4ML sandbox bridge extension
        cmd.extend(["--no-builtin-tools"])
        cmd.extend(["-e", str(self._agent4ml_extension_path())])
        # 决策 5：不传 --system-prompt（MVP 靠 _build_prompt 在 user message 塞 context）
        cmd.extend(self._config.extra_args)
        return cmd

    def _build_env(self, request: SpecialistRequest) -> dict[str, str] | None:
        """透传凭证 env vars + Agent4ML sandbox MCP endpoint（决策 1 + 决策 3）。

        决策 3：凭证 env 国内 provider 优先（DeepSeek/Kimi/MiniMax/Xiaomi 靠前）。
        决策 1：传 AGENT4ML_SANDBOX_MCP_ENDPOINT（sandbox_id 绑定）给 pi extension。
        返 merge 后的 env（父 env + auth vars 覆盖），保证 PATH/HOME 等基础 env 可用。
        """
        auth_env: dict[str, str] = {}
        # 国内 provider 优先（决策 3，便宜优先）
        pi_env_vars = [
            "DEEPSEEK_API_KEY",
            "KIMI_API_KEY",
            "MINIMAX_API_KEY",
            "XIAOMI_API_KEY",
            "ZAI_API_KEY",
            # 国外大厂
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            # 聚合/其他
            "OPENROUTER_API_KEY",
            "GROQ_API_KEY",
            "XAI_API_KEY",
            "MISTRAL_API_KEY",
            "TOGETHER_API_KEY",
        ]
        for var in pi_env_vars:
            val = os.getenv(var)
            if val:
                auth_env[var] = val

        # PI_CODING_AGENT_DIR（自定义配置目录）
        pi_dir = os.getenv("PI_CODING_AGENT_DIR")
        if pi_dir:
            auth_env["PI_CODING_AGENT_DIR"] = pi_dir

        # 决策 1：传 Agent4ML SpecialistMcpServer endpoint（sandbox_id 绑定）
        if request.sandbox_id:
            auth_env["AGENT4ML_SANDBOX_MCP_ENDPOINT"] = self._resolve_mcp_endpoint(
                request.sandbox_id
            )

        if not auth_env:
            return None
        # merge：父进程 env + auth vars 覆盖（保证 PATH/HOME 等基础 env 可用）
        return {**os.environ, **auth_env}

    def _build_prompt(self, request: SpecialistRequest) -> str:
        """构造给 pi 的 prompt（决策 5：MVP 不定制 system prompt）。

        pi 用自己的默认 system prompt（通用 coding agent）。
        Agent4ML 在 user message 里塞 goal + context_summary + success_criteria + 三段输出格式要求。
        """
        parts = [request.goal]
        if request.context_summary:
            parts.append(f"\n\n## Context\n{request.context_summary}")
        if request.success_criteria:
            parts.append(f"\n\n## Success Criteria\n{request.success_criteria}")
        parts.append(
            "\n\n## Output Format\n"
            "After completing the task, summarize in three sections:\n"
            "## What You Did\n"
            "- Files changed (paths + brief description)\n"
            "- Commands run\n"
            "## Success\n"
            "- Whether success criteria are met (yes/no/partial)\n"
            "- Evidence (test results, file existence, etc.)\n"
            "## Gaps\n"
            "- Any incomplete parts or blockers\n"
            "- Next steps if incomplete"
        )
        return "".join(parts)

    def _extract_usage(
        self, usage_data: dict[str, Any]
    ) -> TokenUsage | None:
        """从 pi agent_end 事件的 usage 字段提取 TokenUsage。"""
        if not usage_data:
            return None
        try:
            tokens = usage_data.get("tokens") or {}
            return TokenUsage(
                prompt_tokens=int(tokens.get("input", 0)),
                completion_tokens=int(tokens.get("output", 0)),
                total_tokens=int(tokens.get("total", 0)),
            )
        except (TypeError, ValueError):
            return None

    def _resolve_mcp_endpoint(self, sandbox_id: str) -> str:
        """解析 Agent4ML SpecialistMcpServer endpoint（sandbox_id 绑定）。

        MVP：返回 SpecialistMcpServer 的 stdio 启动命令（pi extension 通过此命令连接）。
        块 D3：若 sandbox_provider 有 SandboxInfo，追加 --sandbox-url + --sandbox-root。
        """
        parts = [
            "python",
            "-m",
            "agent4ml.backend.agents.multiagent.mcp.specialist_mcp_server",
            "--sandbox-id",
            sandbox_id,
        ]
        append_sandbox_url_args(parts, self._sandbox_provider, sandbox_id)
        return " ".join(parts)

    def _agent4ml_extension_path(self) -> str:
        """Agent4ML sandbox bridge extension 路径（随 Agent4ML 安装）。

        决策 1：pi extension 注册 8 工具转发到 Agent4ML SpecialistMcpServer。
        extension 文件在 P4 batch 创建（当前路径返回预期位置，extension 不存在时
        pi 会报 extension_error，由 _run_rpc_session 捕获为 SpecialistCrashError）。
        """
        from agent4ml.backend.agents.multiagent import __path__ as pkg_path

        import pathlib

        return str(
            pathlib.Path(pkg_path[0])
            / "extensions"
            / "pi-sandbox-bridge"
            / "index.ts"
        )
