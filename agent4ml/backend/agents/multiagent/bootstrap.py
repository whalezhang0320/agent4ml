"""Multi-Agent bootstrap — 装配 specialist + 凭证检测 + metrics + 注入 CapabilityRegistry。

设计（spec.md bootstrap Requirement + design.md §2）:
- 反射加载 specialist（config.specialists.use）
- 凭证检测（缺失 → specialist disabled，tool 不注册）
- SubagentProvider 构造（agent_factory 注入）
- metrics store 构造
- OrchestrationMiddleware 构造
- 动态生成 specialist tools
- enabled=false 时不装配（lead agent 行为不变）
- 凭证缺失 warn 提示安装步骤（Bug C 修复，设计文档 46 §4.4）
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.tools import BaseTool

from agent4ml.backend.agents.multiagent.config import MultiAgentConfig
from agent4ml.backend.agents.multiagent.middleware import OrchestrationMiddleware
from agent4ml.backend.agents.multiagent.metrics import MultiAgentMetricsStore
from agent4ml.backend.agents.multiagent.registry import SpecialistRegistry
from agent4ml.backend.agents.multiagent.runtimes.subagent_runtime import SubagentRuntime
from agent4ml.backend.agents.multiagent.tools import (
    make_specialist_tool,
    make_subagent_tool,
)

logger = logging.getLogger(__name__)


def _warn_specialist_disabled(name: str, reason: str) -> None:
    """启动时 warn 提示用户如何启用该 specialist（不阻塞主流程）。

    Bug C 修复（设计文档 46 §4.4）：
    - pi/codex/claude 凭证缺失 → warning 级别 + 安装步骤
    - subagent 失败 → error 级别（应是 bug，subagent 应零配置可用）
    """
    if name == "pi":
        logger.warning(
            "[PiSpecialist] disabled: %s\n"
            "To enable, install Pi CLI and set any API key:\n"
            "  npm install -g @earendil-works/pi-coding-agent\n"
            "  Set any of: ANTHROPIC_API_KEY, OPENAI_API_KEY, DEEPSEEK_API_KEY, "
            "KIMI_API_KEY, MINIMAX_API_KEY, etc.\n"
            "Or configure multiagent.specialists.pi.provider in config.yaml",
            reason,
        )
    elif name == "codex":
        logger.warning(
            "[CodexSpecialist] disabled: %s\n"
            "To enable, install Codex CLI and login:\n"
            "  npm install -g @openai/codex\n"
            "  codex login\n"
            "Or set CODEX_AUTH_PATH env var pointing to auth.json",
            reason,
        )
    elif name == "claude":
        logger.warning(
            "[ClaudeCodeSpecialist] disabled: %s\n"
            "To enable, install Claude Code CLI and login:\n"
            "  npm install -g @anthropic/claude-code\n"
            "  claude /login\n"
            "Or set CLAUDE_CODE_OAUTH_TOKEN / ANTHROPIC_AUTH_TOKEN env var",
            reason,
        )
    elif name == "subagent":
        logger.error(
            "[SubagentSpecialist] disabled: %s\n"
            "This is a bug — subagent should be zero-config. "
            "Check that agent_factory is injected in bootstrap.",
            reason,
        )
    else:
        logger.warning(
            "[Specialist:%s] disabled: %s", name, reason
        )


@dataclass(frozen=True)
class MultiAgentSetup:
    """setup_multiagent 结果——注入 CapabilityRegistry + factory。"""

    specialist_registry: SpecialistRegistry | None
    subagent_provider: SubagentRuntime | None
    metrics_store: MultiAgentMetricsStore | None
    orchestration_middleware: OrchestrationMiddleware | None
    specialist_tools: tuple[BaseTool, ...]
    # L2 evolution layer (None when config.l2.enabled=false)
    l2_setup: Any | None = None


_EMPTY_SETUP = MultiAgentSetup(
    specialist_registry=None,
    subagent_provider=None,
    metrics_store=None,
    orchestration_middleware=None,
    specialist_tools=(),
    l2_setup=None,
)


def _load_specialist(
    name: str,
    config: MultiAgentConfig,
    agent_factory: Callable[[], Any] | None = None,
) -> tuple[Any, Any, Any] | None:
    """反射加载 specialist + 凭证检测 + 匹配 summarizer。

    返 None 表示 specialist disabled（凭证缺失或未知 name）。
    """
    if name == "pi":
        from agent4ml.backend.agents.multiagent.installer.pi_installer import (
            PiInstaller,
        )
        from agent4ml.backend.agents.multiagent.credentials.pi_credential import (
            PiCredentialProvider,
        )

        # 决策 2：确保 pi 已装（后台安装不阻塞）
        installer = PiInstaller(
            auto_install=config.specialists_pi_auto_install
        )
        if not installer.ensure_installed():
            return None  # pi 不可用，disabled（后台安装中或不可装）

        # 决策 3：双轨凭证解析（config 优先 + env 兜底）
        cred_provider = PiCredentialProvider(
            config_provider=config.specialists_pi_provider or None,
            config_api_key=config.specialists_pi_api_key or None,
        )
        cred = cred_provider.get_credential()
        if cred is None:
            return None  # 凭证缺失，disabled

        # 决策 1 + 决策 5：加载 PiSpecialist（PiRuntime 内部 --no-builtin-tools + extension）
        from agent4ml.backend.agents.multiagent.runtimes.pi_runtime import (
            PiRuntime,
            PiRuntimeConfig,
        )
        from agent4ml.backend.agents.multiagent.specialists.pi_specialist import (
            PiSpecialist,
        )
        from agent4ml.backend.agents.multiagent.summarizers.context.pi_context_summarizer import (
            PiContextSummarizer,
        )
        from agent4ml.backend.agents.multiagent.summarizers.result.pi_result_summarizer import (
            PiResultSummarizer,
        )

        runtime_config = PiRuntimeConfig(
            provider=cred.provider,
            model=config.specialists_pi_model or None,
            thinking_level=config.specialists_pi_thinking_level,
        )
        return (
            PiSpecialist(runtime=PiRuntime(config=runtime_config), credential=cred),
            PiContextSummarizer(),
            PiResultSummarizer(),
        )

    if name == "codex":
        from agent4ml.backend.agents.multiagent.credentials.codex_credential import (
            CodexCredentialProvider,
        )
        cred = CodexCredentialProvider().get_credential()
        if cred is None:
            return None
        from agent4ml.backend.agents.multiagent.specialists.codex_specialist import (
            CodexSpecialist,
        )
        from agent4ml.backend.agents.multiagent.summarizers.context.codex_context_summarizer import (
            CodexContextSummarizer,
        )
        from agent4ml.backend.agents.multiagent.summarizers.result.codex_result_summarizer import (
            CodexResultSummarizer,
        )
        return CodexSpecialist(), CodexContextSummarizer(), CodexResultSummarizer()

    if name == "claude":
        from agent4ml.backend.agents.multiagent.credentials.claude_credential import (
            ClaudeCredentialProvider,
        )
        cred = ClaudeCredentialProvider().get_credential()
        if cred is None:
            return None
        from agent4ml.backend.agents.multiagent.specialists.claude_code_specialist import (
            ClaudeCodeSpecialist,
        )
        from agent4ml.backend.agents.multiagent.summarizers.context.claude_code_context_summarizer import (
            ClaudeCodeContextSummarizer,
        )
        from agent4ml.backend.agents.multiagent.summarizers.result.claude_code_result_summarizer import (
            ClaudeCodeResultSummarizer,
        )
        return (
            ClaudeCodeSpecialist(),
            ClaudeCodeContextSummarizer(),
            ClaudeCodeResultSummarizer(),
        )

    if name == "subagent":
        from agent4ml.backend.agents.multiagent.runtimes.subagent_runtime import (
            SubagentRuntime,
        )
        from agent4ml.backend.agents.multiagent.specialists.subagent_specialist import (
            SubagentSpecialist,
        )
        from agent4ml.backend.agents.multiagent.summarizers.context.self_copy_context_summarizer import (
            SelfCopyContextSummarizer,
        )
        from agent4ml.backend.agents.multiagent.summarizers.result.self_copy_result_summarizer import (
            SelfCopyResultSummarizer,
        )
        runtime = SubagentRuntime(agent_factory=agent_factory) if agent_factory else SubagentRuntime()
        return (
            SubagentSpecialist(runtime=runtime),
            SelfCopyContextSummarizer(),
            SelfCopyResultSummarizer(),
        )

    return None


def setup_multiagent(
    config: MultiAgentConfig,
    agent_factory: Callable[[], Any] | None = None,
) -> MultiAgentSetup:
    """装配 multi-agent orchestration。

    enabled=false → 返空 setup（lead agent 行为不变）。
    enabled=true → 反射加载 specialist + 凭证检测 + metrics + middleware + tools。
    """
    if not config.enabled:
        return _EMPTY_SETUP

    metrics = MultiAgentMetricsStore(config.metrics_db_path)

    # L2 evolution layer assembly (enabled=false -> l2_setup=None, L1 behavior unchanged)
    from agent4ml.backend.agents.multiagent.evolution.bootstrap import setup_l2
    l2_setup = setup_l2(config, metrics)

    orch_mw = OrchestrationMiddleware(
        metrics_store=metrics,
        l2_trigger_middleware=l2_setup.l2_trigger_middleware if l2_setup else None,
        budget_guard=l2_setup.budget_guard if l2_setup else None,
    )
    specialist_registry = SpecialistRegistry()
    tools: list[BaseTool] = []

    for name in config.specialists_use:
        loaded = _load_specialist(name, config, agent_factory=agent_factory)
        if loaded is None:
            _warn_specialist_disabled(name, "credential missing or load failed")
            continue
        specialist, ctx_summarizer, result_summarizer = loaded
        specialist_registry.register(specialist)
        tools.append(
            make_specialist_tool(
                name,
                specialist,
                ctx_summarizer,
                result_summarizer,
                max_steps=config.max_steps,
                timeout_seconds=config.timeout_seconds,
                version_dag=l2_setup.version_dag if l2_setup else None,
                budget_guard=l2_setup.budget_guard if l2_setup else None,
            )
        )

    subagent_provider: SubagentRuntime | None = None
    if agent_factory is not None and "subagent" not in config.specialists_use:
        # Only create standalone delegate_to_subagent tool if "subagent" not already a specialist
        subagent_provider = SubagentRuntime(agent_factory=agent_factory)
        from agent4ml.backend.agents.multiagent.summarizers.context.self_copy_context_summarizer import (
            SelfCopyContextSummarizer,
        )
        from agent4ml.backend.agents.multiagent.summarizers.result.self_copy_result_summarizer import (
            SelfCopyResultSummarizer,
        )
        tools.append(
            make_subagent_tool(
                subagent_provider,
                SelfCopyContextSummarizer(),
                SelfCopyResultSummarizer(),
                max_steps=config.subagent_max_steps,
                timeout_seconds=config.subagent_timeout_seconds,
            )
        )

    # Start L2 daemon thread if L2 enabled
    if l2_setup is not None:
        l2_setup.worker.start()

    return MultiAgentSetup(
        specialist_registry=specialist_registry,
        subagent_provider=subagent_provider,
        metrics_store=metrics,
        orchestration_middleware=orch_mw,
        specialist_tools=tuple(tools),
        l2_setup=l2_setup,
    )
