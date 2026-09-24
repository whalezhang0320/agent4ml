from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from agent4ml.backend.agents.agent_tools.available import get_available_tools
from agent4ml.backend.agents.capabilities.registry import CapabilityRegistry
from agent4ml.backend.agents.leader.prompts import apply_prompt_template
from agent4ml.backend.agents.middlewares.dangling_tool_call_middleware import (
    DanglingToolCallMiddleware,
)
from agent4ml.backend.agents.middlewares.evidence_middleware import EvidenceMiddleware
from agent4ml.backend.agents.middlewares.help_request_middleware import (
    HelpRequestMiddleware,
)
from agent4ml.backend.agents.middlewares.loop_detection_middleware import (
    LoopDetectionMiddleware,
)
from agent4ml.backend.agents.middlewares.stall_detection_middleware import (
    StallDetectionMiddleware,
)
from agent4ml.backend.agents.middlewares.reflection_middleware import (
    LightReflectionStrategy,
    ReflectionMiddleware,
    SufficiencyStrategy,
)
from agent4ml.backend.agents.middlewares.report_middleware import ReportMiddleware
from agent4ml.backend.agents.middlewares.run_journal_middleware import RunJournalMiddleware
from agent4ml.backend.agents.middlewares.system_context_middleware import SystemContextMiddleware
from agent4ml.backend.agents.middlewares.title_middleware import TitleMiddleware
from agent4ml.backend.agents.middlewares.todo_middleware import TodoMiddleware
from agent4ml.backend.agents.middlewares.tool_call_middleware import ToolCallMiddleware
from agent4ml.backend.agents.runtime.checkpointer import get_checkpointer
from agent4ml.backend.agents.state.types import ThreadState


def _safe_get_specialist_registry(registry: Any) -> Any | None:
    """从 CapabilityRegistry 取 specialist_registry，缺失返 None（不抛 CapabilityMissingError）。

    Bug B 修复：make_lead_agent 调 apply_prompt_template 时传 specialist_registry。
    缺失（multiagent disabled 或无 specialist 注册）时返 None，
    apply_prompt_template 不注入 <specialist_routing> 段（保护 prompt caching）。
    """
    try:
        return registry.get_specialist_registry()
    except Exception:
        return None


def _build_middlewares(
    expert_mode: bool,
    model: BaseChatModel | None = None,
    context_governance: Any = None,
    summarize_model: Any = None,
    sandbox_provider: Any = None,
    artifact_server: Any = None,
    mcp_audit_middleware: Any = None,
    skill_injection_middleware: Any = None,
    skill_metrics_middleware: Any = None,
    orchestration_middleware: Any = None,
    memory_provider: Any = None,
    memory_config: Any = None,
    memory_worker: Any = None,
) -> list:
    """全模式全挂 middleware，参数化控制行为差异。

    default (expert_mode=False): 温和参数——Todo 不强制完成、Reflection 不 jump、
        Report 不自动合成。靠 prompt 引导模型自判深度。
    expert (expert_mode=True): 激进参数——Todo 强制完成、Reflection 充分性 jump、
        Report after_agent 自动合成。

    治理层（context_governance）挂载顺序见 builder.build_governance_middlewares。
    挂载顺序：治理层（公共3 + StrategyMiddleware） → SystemContext → SkillInjection（条件挂）
    → SkillMetrics（条件挂）→ Title → RunJournal → MCP Audit（条件挂）→ Sandbox（条件挂）
    → LoopDetection → ToolCall → Evidence → Todo → Reflection → Report。
    """
    middlewares: list = []
    if context_governance is not None:
        from agent4ml.backend.agents.context_engineering.builder import (
            build_governance_middlewares,
        )

        middlewares.extend(build_governance_middlewares(context_governance, model=model, summarize_model=summarize_model))
    middlewares.extend([
        SystemContextMiddleware(),
    ])
    if skill_injection_middleware is not None:
        middlewares.append(skill_injection_middleware)
    if skill_metrics_middleware is not None:
        middlewares.append(skill_metrics_middleware)
    # F3 改造：SkillActivationMiddleware before_model 主动建议相关 skill
    # 参数化挂载：skills 启用时（skill_injection_middleware 非空）挂载
    if skill_injection_middleware is not None:
        from agent4ml.backend.agents.middlewares.skill_activation_middleware import (
            SkillActivationMiddleware,
        )
        middlewares.append(SkillActivationMiddleware())
    middlewares.extend([
        TitleMiddleware(),
        RunJournalMiddleware(),
    ])
    if mcp_audit_middleware is not None:
        middlewares.append(mcp_audit_middleware)
    if sandbox_provider is not None:
        from agent4ml.backend.agents.middlewares.sandbox_middleware import (
            SandboxMiddleware,
        )

        from pathlib import Path
        sandbox_root = str(Path.cwd() / ".agent4ml" / "sandbox" / "aio_docker")
        middlewares.append(SandboxMiddleware(
            provider=sandbox_provider,
            artifact_server=artifact_server,
            sandbox_root=sandbox_root,
        ))
    # L4 Memory: Sandbox 后,HelpRequest/ToolCall 前(记忆引用 sandbox 结果,不进 tool pairing)
    if memory_provider is not None:
        from agent4ml.backend.agents.middlewares.memory_recall_middleware import MemoryMiddleware

        middlewares.append(MemoryMiddleware(
            memory_provider=memory_provider,
            enable_recall=memory_config.enable_recall if memory_config else True,
            enable_extract=memory_config.enable_extract if memory_config else False,
            token_budget=memory_config.token_budget if memory_config else 2000,
        ))
    # L5 Memory Consolidation: MemoryMiddleware 后(HelpRequest 前)
    if memory_worker is not None:
        from agent4ml.backend.agents.middlewares.memory_consolidation_middleware import (
            MemoryConsolidationMiddleware,
        )
        trigger_n = (
            memory_config.phase2.get("trigger_every_n_turns", 10)
            if memory_config else 10
        )
        middlewares.append(MemoryConsolidationMiddleware(
            worker=memory_worker,
            trigger_every_n_turns=trigger_n,
        ))
    middlewares.extend([
        HelpRequestMiddleware(),
        DanglingToolCallMiddleware(),
        # LoopDetectionMiddleware 已移除——用户要求取消循环上限约束。
        # 原配置：after_model 检测近 10 条消息同 (tool, args_hash) ≥3 → 清 tool_calls + jump model。
        # 如需恢复，取消下行注释 + 确保 import 存在。
        # LoopDetectionMiddleware(),
        ToolCallMiddleware(),
    ])
    if orchestration_middleware is not None:
        middlewares.append(orchestration_middleware)
    middlewares.extend([
        EvidenceMiddleware(),
        StallDetectionMiddleware(),
        TodoMiddleware(enforce_completion=expert_mode),
        ReflectionMiddleware(
            strategy=SufficiencyStrategy(llm=model) if expert_mode else LightReflectionStrategy(),
            llm=model,
        ),
    ])
    if model is not None:
        middlewares.append(ReportMiddleware(model, auto_synthesize=expert_mode))
    return middlewares


def make_lead_agent(
    expert_mode: bool = False,
    capability_registry: CapabilityRegistry | None = None,
    middleware_manager: Any = None,
    runnable_config: RunnableConfig | None = None,
    context_governance: Any = None,
    sandbox_provider: Any = None,
    artifact_server: Any = None,
    mcp_audit_middleware: Any = None,
    skill_injection_middleware: Any = None,
    skill_metrics_middleware: Any = None,
    specialist_tools: list[BaseTool] | None = None,
    orchestration_middleware: Any = None,
    memory_provider: Any = None,
    memory_config: Any = None,
    memory_worker: Any = None,
) -> Any:
    """App-layer factory: expert_flag 参数化装配 graph。

    - middleware: _build_middlewares(expert_mode) 全挂参数化
    - tools: get_available_tools(groups=...) 按 expert 选 core / core+deferred
    - system_prompt: apply_prompt_template(expert_mode, specialist_registry=...)
      — specialist_registry 非空时条件注入 <specialist_routing> 段（Bug B 修复）
    - checkpointer: get_checkpointer() 单例，thread_id 跨轮 + 跨模式保留 state

    Registry MUST store BaseChatModel / BaseTool instances directly.
    """
    from agent4ml.backend.agents.leader.agent import LeaderAgent

    if runnable_config is not None and "configurable" in runnable_config:
        registry = runnable_config["configurable"].get("capability_registry")
    else:
        registry = capability_registry
        runnable_config = {"configurable": {"expert_mode": expert_mode, "capability_registry": registry}}

    if registry is None:
        raise ValueError("capability_registry is required (in configurable or as param)")

    configurable = runnable_config.get("configurable", {})
    # runnable_config 透传的 expert_mode 优先，否则用参数
    resolved_expert = bool(configurable.get("expert_mode", expert_mode))

    model = registry.get_model("researcher")

    # summarize_model 独立取：config.params.summarize_model 配名 → registry 取独立 model；否则 None（fallback research model）
    summarize_model = None
    if context_governance is not None:
        sm_name = (getattr(context_governance, "params", None) or {}).get("summarize_model")
        if sm_name:
            try:
                summarize_model = registry.get_model(sm_name)
            except Exception:
                summarize_model = None

    # tool groups: default=core, expert=core+deferred
    groups = ["core", "deferred"] if resolved_expert else ["core"]
    tools: list[BaseTool] = []
    for tool in get_available_tools(groups=groups):
        if isinstance(tool, BaseTool) and tool not in tools:
            tools.append(tool)
    # registry 额外注册的工具（如 web_search_mcp 别名）保留
    for tool in registry.tools.values():
        if isinstance(tool, BaseTool) and tool not in tools:
            tools.append(tool)
    # multiagent specialist tools（delegate_to_*）
    if specialist_tools:
        for tool in specialist_tools:
            if isinstance(tool, BaseTool) and tool not in tools:
                tools.append(tool)

    return LeaderAgent(
        graph=create_agent(
            model=model,
            tools=tools or None,
            middleware=_build_middlewares(resolved_expert, model, context_governance, summarize_model, sandbox_provider, artifact_server, mcp_audit_middleware, skill_injection_middleware, skill_metrics_middleware, orchestration_middleware, memory_provider, memory_config, memory_worker),
            system_prompt=apply_prompt_template(
                expert_mode=resolved_expert,
                specialist_registry=_safe_get_specialist_registry(registry),
                skills_enabled=skill_injection_middleware is not None,
            ),
            state_schema=ThreadState,
            checkpointer=get_checkpointer(),
        ),
        capability_registry=registry,
    )
