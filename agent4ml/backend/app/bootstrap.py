from __future__ import annotations

import random
import shutil
import string
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel

from agent4ml.backend.agents.artifacts.local_store import LocalArtifactStore
from agent4ml.backend.agents.capabilities.registry import CapabilityRegistry
from agent4ml.backend.agents.config.loader import load_config
from agent4ml.backend.agents.memory.bootstrap import (
    get_memory_worker,
    shutdown_memory_worker,
    start_memory_worker,
)
from agent4ml.backend.agents.memory.config import set_memory_config
from agent4ml.backend.agents.config.provider_config import ProviderConfig, select_provider_config
from agent4ml.backend.agents.config.schema import AppConfig
from agent4ml.backend.agents.journal.events import utc_now_iso
from agent4ml.backend.agents.journal.run_journal import RunJournal
from agent4ml.backend.agents.leader.agent import AgentRunResult, LeaderAgent
from agent4ml.backend.agents.leader.factory import make_lead_agent
from agent4ml.backend.agents.reporting.markdown_reporter import MarkdownReporter
from agent4ml.backend.agents.runtime.run_manager import RunManager
from agent4ml.backend.agents.agent_tools.available import get_available_tools, select_search_tool
from agent4ml.backend.agents.multiagent.bootstrap import MultiAgentSetup, setup_multiagent
from agent4ml.backend.agents.multiagent.config import load_multiagent_config

_PROJECT_ROOT = Path(__file__).parents[3]
_CST = timezone(timedelta(hours=8))


def _resolve_relative_paths(config: AppConfig) -> AppConfig:
    """把 config 里相对路径锚到 _PROJECT_ROOT——与 logs_root 同款处理。

    目前覆盖 ``context_governance.params.externalize_dir``（默认 ``.agent4ml/externalized``）。
    之前没锚定，ExternalizerExecutor 直接 ``os.makedirs(self._dir)`` 按进程 CWD 解析——
    用户从 PowerShell 启动 ``agent4ml``（默认 CWD=用户家目录）时，外化文件全部写到了
    ``C:\\Users\\<user>\\.agent4ml\\externalized\\``，与项目目录下的 ``.agent4ml/externalized``
    分家，用户在项目目录看不到任何外化记录（D12 现场定位）。
    """
    params = dict(config.context_governance.params)
    ext_dir = params.get("externalize_dir", ".agent4ml/externalized")
    p = Path(ext_dir)
    if not p.is_absolute():
        p = (_PROJECT_ROOT / p).resolve()
    params["externalize_dir"] = str(p)
    # L4 memory.storage_path 锚定 _PROJECT_ROOT（01 D12，与 externalize_dir 同款）
    memory_path = Path(config.memory.storage_path)
    if not memory_path.is_absolute():
        memory_path = (_PROJECT_ROOT / memory_path).resolve()
    config = replace(config, memory=replace(config.memory, storage_path=str(memory_path)))
    return replace(
        config,
        context_governance=replace(config.context_governance, params=params),
    )


def _make_thread_id() -> str:
    ts = datetime.now(_CST).strftime("%Y%m%dT%H%M%S")
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"thread-{ts}-{suffix}"


def _build_chat_model(config: ProviderConfig) -> BaseChatModel:
    config.require_api_key()
    if config.provider == "deepseek":
        from langchain_deepseek import ChatDeepSeek
        return ChatDeepSeek(model=config.model, api_key=config.api_key)
    if config.provider in ("openai", "qwen"):
        from langchain_openai import ChatOpenAI
        kwargs: dict[str, Any] = {"model": config.model, "api_key": config.api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOpenAI(**kwargs)
    raise ValueError(f"unsupported provider: {config.provider}")


def _check_node_available() -> bool:
    return shutil.which("npx") is not None


@dataclass
class AppRuntime:
    config: AppConfig
    capability_registry: CapabilityRegistry
    run_manager: RunManager
    researcher_model_name: str
    thread_id: str
    thread_dir: Path
    thread_journal: RunJournal
    leader_agent: LeaderAgent
    mcp_manager: Any = None
    artifact_server: Any = None
    skill_manager: Any = None
    multiagent_setup: MultiAgentSetup | None = None

    def run_question(
        self,
        question: str,
        thread_id: str | None = None,
        user_id: str | None = "default-user",
        run_id: str | None = None,
    ) -> AgentRunResult:
        effective_thread_id = thread_id or self.thread_id
        context = self.run_manager.create_run(
            thread_id=effective_thread_id,
            user_id=user_id,
            run_id=run_id,
            model_name=self.researcher_model_name,
            thread_dir=self.thread_dir,
        )
        self.run_manager.mark_running(context.run_id)
        try:
            result = self.leader_agent.run(question, context)
            self.run_manager.mark_success(context.run_id)
            return result
        except Exception as exc:
            self.run_manager.mark_failed(context.run_id, str(exc))
            raise

    def switch_expert_mode(self, expert_mode: bool) -> AppRuntime:
        """切换 expert 模式，精准重建受影响部分，保留 thread 连续性。

        重建：config + run_manager + leader_agent（依赖 expert_mode 编译参数）。
        保留：thread_id / thread_dir / thread_journal / capability_registry /
        researcher_model_name（checkpointer state 跨模式连续，MCP/models 不重载）。

        返回新 AppRuntime 实例（不可变语义），CLI 用 runtime = runtime.switch_expert_mode(...)。
        """
        new_config = load_config(expert_mode=expert_mode)
        logs_root = Path(new_config.runtime.logs_root)
        if not logs_root.is_absolute():
            logs_root = _PROJECT_ROOT / logs_root
        new_config = replace(
            new_config,
            runtime=replace(new_config.runtime, logs_root=str(logs_root)),
        )
        # 锚定 externalize_dir 等治理层相对路径到项目根（同 logs_root 处理）
        new_config = _resolve_relative_paths(new_config)
        # 必须传 context_governance——否则 _build_middlewares 看到 None 会跳过整个
        # 治理层（StrategyMiddleware 不挂），切换 expert 后 budget/fraction/压缩全部失效，
        # 与 D12 "minimal 未注册" 故障现象相同。
        new_leader = make_lead_agent(
            expert_mode=expert_mode,
            capability_registry=self.capability_registry,
            context_governance=new_config.context_governance,
            sandbox_provider=getattr(self.capability_registry, "sandbox_provider", None),
            artifact_server=self.artifact_server,
            mcp_audit_middleware=self.mcp_manager.get_audit_middleware() if self.mcp_manager else None,
            skill_injection_middleware=self.skill_manager.get_injection_middleware() if self.skill_manager else None,
            skill_metrics_middleware=self.skill_manager.get_metrics_middleware() if self.skill_manager else None,
            specialist_tools=list(self.multiagent_setup.specialist_tools) if self.multiagent_setup else None,
            orchestration_middleware=self.multiagent_setup.orchestration_middleware if self.multiagent_setup else None,
            memory_provider=getattr(self.capability_registry, "memory_provider", None),
            memory_config=self.config.memory,
            memory_worker=get_memory_worker(),
        )
        self.thread_journal.append("mode.switched", {
            "expert_mode": expert_mode,
            "thread_id": self.thread_id,
        })
        return AppRuntime(
            config=new_config,
            capability_registry=self.capability_registry,
            run_manager=RunManager(new_config),
            researcher_model_name=self.researcher_model_name,
            thread_id=self.thread_id,
            thread_dir=self.thread_dir,
            thread_journal=self.thread_journal,
            leader_agent=new_leader,
            mcp_manager=self.mcp_manager,
            artifact_server=self.artifact_server,
            skill_manager=self.skill_manager,
            multiagent_setup=self.multiagent_setup,
        )

    def reload_mcp_tools(self) -> AppRuntime:
        """MCP 工具变更后重建 LeaderAgent graph。

        复用 switch_expert_mode 模式：重建 leader_agent（新工具注入），保留 thread_id /
        thread_dir / thread_journal / capability_registry，checkpointer state 跨重建连续。
        同步完成（<1s），下轮可用（当前轮用旧 graph 跑完）。
        """
        expert_mode = self.config.runtime.expert_mode if hasattr(self.config.runtime, "expert_mode") else False
        new_leader = make_lead_agent(
            expert_mode=expert_mode,
            capability_registry=self.capability_registry,
            context_governance=self.config.context_governance,
            sandbox_provider=getattr(self.capability_registry, "sandbox_provider", None),
            artifact_server=self.artifact_server,
            mcp_audit_middleware=self.mcp_manager.get_audit_middleware() if self.mcp_manager else None,
            skill_injection_middleware=self.skill_manager.get_injection_middleware() if self.skill_manager else None,
            skill_metrics_middleware=self.skill_manager.get_metrics_middleware() if self.skill_manager else None,
            specialist_tools=list(self.multiagent_setup.specialist_tools) if self.multiagent_setup else None,
            orchestration_middleware=self.multiagent_setup.orchestration_middleware if self.multiagent_setup else None,
            memory_provider=getattr(self.capability_registry, "memory_provider", None),
            memory_config=self.config.memory,
            memory_worker=get_memory_worker(),
        )
        self.thread_journal.append("mcp.tools_reloaded", {"thread_id": self.thread_id})
        return AppRuntime(
            config=self.config,
            capability_registry=self.capability_registry,
            run_manager=self.run_manager,
            researcher_model_name=self.researcher_model_name,
            thread_id=self.thread_id,
            thread_dir=self.thread_dir,
            thread_journal=self.thread_journal,
            leader_agent=new_leader,
            mcp_manager=self.mcp_manager,
            artifact_server=self.artifact_server,
            skill_manager=self.skill_manager,
            multiagent_setup=self.multiagent_setup,
        )

    def switch_model(self, provider: str, model: str | None = None) -> AppRuntime:
        """热切换 LLM provider/model。重建 researcher+reporter model + capability_registry
        + leader_agent，保留 thread_id / thread_dir / thread_journal / mcp_manager /
        artifact_server / skill_manager / sandbox_provider（checkpointer state 跨切换连续）。

        等价于 CLI ``--provider X --model Y`` 重启，但不丢 thread。同 ``switch_expert_mode``
        不可变语义——返回新 AppRuntime，CLI 用 runtime = runtime.switch_model(...)。

        provider 必须是 MODEL_PROVIDERS 里 enabled 的项；model=None 用 provider 默认 model。
        单 provider 模式（不走 FallbackChatModel 路由链），reporter = researcher。
        """
        from agent4ml.backend.agents.config.model_router import ModelRouter

        router = ModelRouter()
        new_model = router.build_single(provider, model)  # 校验 provider + api_key，失败抛 ProviderConfigError
        new_reporter = new_model
        new_registry = CapabilityRegistry(
            models={"researcher": new_model, "reporter": new_reporter},
            tools=self.capability_registry.tools,
            reporter=self.capability_registry.reporter,
            artifact_store=self.capability_registry.artifact_store,
            sandbox_provider=self.capability_registry.sandbox_provider,
            skill_store=self.capability_registry.skill_store,
            specialist_registry=self.capability_registry.specialist_registry,
            subagent_provider=self.capability_registry.subagent_provider,
            memory_provider=self.capability_registry.memory_provider,
        )
        expert_mode = self.config.runtime.expert_mode
        new_leader = make_lead_agent(
            expert_mode=expert_mode,
            capability_registry=new_registry,
            context_governance=self.config.context_governance,
            sandbox_provider=getattr(new_registry, "sandbox_provider", None),
            artifact_server=self.artifact_server,
            mcp_audit_middleware=self.mcp_manager.get_audit_middleware() if self.mcp_manager else None,
            skill_injection_middleware=self.skill_manager.get_injection_middleware() if self.skill_manager else None,
            skill_metrics_middleware=self.skill_manager.get_metrics_middleware() if self.skill_manager else None,
            specialist_tools=list(self.multiagent_setup.specialist_tools) if self.multiagent_setup else None,
            orchestration_middleware=self.multiagent_setup.orchestration_middleware if self.multiagent_setup else None,
            memory_provider=getattr(self.capability_registry, "memory_provider", None),
            memory_config=self.config.memory,
            memory_worker=get_memory_worker(),
        )
        self.thread_journal.append("model.switched", {
            "provider": provider,
            "model": model or "default",
            "thread_id": self.thread_id,
        })
        return AppRuntime(
            config=self.config,
            capability_registry=new_registry,
            run_manager=self.run_manager,
            researcher_model_name=model or provider,
            thread_id=self.thread_id,
            thread_dir=self.thread_dir,
            thread_journal=self.thread_journal,
            leader_agent=new_leader,
            mcp_manager=self.mcp_manager,
            artifact_server=self.artifact_server,
            skill_manager=self.skill_manager,
            multiagent_setup=self.multiagent_setup,
        )


def _load_sandbox_provider(config: AppConfig) -> Any:
    """反射加载 sandbox provider。config.sandbox.use 为空则返 None（Grill #9）。"""
    sandbox_config = config.sandbox
    if not sandbox_config.use:
        return None
    import importlib

    module_path, _, class_name = sandbox_config.use.partition(":")
    module = importlib.import_module(module_path)
    provider_cls = getattr(module, class_name)
    path_mappings = _build_path_mappings(sandbox_config)
    return provider_cls(path_mappings=path_mappings, sandbox_config=sandbox_config)


def _load_memory_provider(config: AppConfig) -> Any:
    """反射加载 memory provider。config.memory.use 为空则返 None。

    照抄 _load_sandbox_provider 反射模式（01 介入点 11.1）。
    config.memory.use="default" 时调 get_memory_provider()（内部 build_default_provider）。
    """
    memory_config = config.memory
    if not memory_config.use:
        return None
    from agent4ml.backend.agents.memory.bootstrap import get_memory_provider

    return get_memory_provider()


def _build_path_mappings(sandbox_config: Any) -> list:
    """从 config 构造 PathMapping 列表。路径锚定 .agent4ml/sandbox/local/（类型分层）。"""
    from agent4ml.backend.agents.sandbox.types import PathMapping

    sandbox_root = _PROJECT_ROOT / ".agent4ml" / "sandbox" / "local"
    mappings = [
        PathMapping("/mnt/agent4ml/user-data/workspace", str(sandbox_root / "workspace")),
        PathMapping("/mnt/agent4ml/user-data/uploads", str(sandbox_root / "uploads")),
        PathMapping("/mnt/agent4ml/user-data/outputs", str(sandbox_root / "outputs")),
    ]
    for mount in sandbox_config.mounts:
        mappings.append(PathMapping(mount.container_path, mount.host_path, mount.read_only))
    return mappings


def _build_evolution_manager(skill_manager: Any, llm: Any, journal: Any) -> Any:
    """建 EvolutionManager（Layer 2a）注入 SkillManager。

    lazy import evolution 模块（避免 skill → evolution → skill 循环）。
    """
    from agent4ml.backend.agents.skill.evolution.focus.ive_focuser import IVEFocuser
    from agent4ml.backend.agents.skill.evolution.eval.programmatic_bridge import (
        ProgrammaticEvalBridge,
    )
    from agent4ml.backend.agents.skill.evolution.gates.score_delta_gate import ScoreDeltaGate
    from agent4ml.backend.agents.skill.evolution.manager import EvolutionManager
    from agent4ml.backend.agents.skill.evolution.mutators.llm_mutator import LLMMutator
    from agent4ml.backend.agents.skill.evolution.triggers.capture_trigger import (
        CaptureTrigger,
    )
    from agent4ml.backend.agents.skill.evolution.triggers.metric_monitor import (
        MetricMonitorTrigger,
    )

    cfg = skill_manager.config
    triggers = [
        MetricMonitorTrigger(
            threshold=cfg.evolve_threshold,
            min_selections=cfg.evolve_min_selections,
            cooldown_turns=cfg.evolve_cooldown_turns,
            llm=llm,
        ),
        CaptureTrigger(),
    ]
    return EvolutionManager(
        store=skill_manager.store,
        triggers=triggers,
        focuser=IVEFocuser(llm=llm),
        mutator=LLMMutator(max_changed_lines=cfg.evolve_mutate_budget, max_steps=cfg.evolve_max_steps, llm=llm),
        eval_bridge=ProgrammaticEvalBridge(),
        gate=ScoreDeltaGate(),
        llm=llm,
        journal=journal,
    )


def _build_eval_layer(skill_manager: Any, llm: Any) -> Any:
    """建 L3 EvalLayer 注入 SkillManager。

    lazy import eval 模块（避免循环依赖）。
    若 EvolutionManager 已装配，替换其 eval_bridge 为 RegistryEvalBridge。
    """
    from agent4ml.backend.agents.skill.eval import EvalLayer
    from agent4ml.backend.agents.skill.eval.analyzers.contract_compiler import ContractCompiler
    from agent4ml.backend.agents.skill.eval.analyzers.response_contract_checker import (
        ResponseContractChecker,
    )
    from agent4ml.backend.agents.skill.eval.analyzers.skill_judgment_analyzer import (
        SkillJudgmentAnalyzer,
    )
    from agent4ml.backend.agents.skill.eval.analyzers.task_quality_judge import (
        TaskQualityJudge,
    )
    from agent4ml.backend.agents.skill.eval.registry import EvalRegistry, RegistryEvalBridge
    from agent4ml.backend.agents.skill.eval.runtime_tracker import RuntimeTracker

    cfg = skill_manager.config.eval_config
    store = skill_manager.store

    checker = ResponseContractChecker(ContractCompiler())
    registry = EvalRegistry(checker)
    bridge = RegistryEvalBridge(registry)

    judgment_analyzer = SkillJudgmentAnalyzer(llm, store) if cfg.judgment_enabled else None
    task_judge = TaskQualityJudge(llm, store) if cfg.task_judge_enabled else None
    runtime_tracker = RuntimeTracker(store, cfg.degradation_delta)

    # 替换 EvolutionManager 的 eval_bridge（若已装配）
    evo = skill_manager.get_evolution_manager()
    if evo is not None:
        evo._eval_bridge = bridge

    return EvalLayer(
        bridge=bridge,
        judgment_analyzer=judgment_analyzer,
        task_judge=task_judge,
        runtime_tracker=runtime_tracker,
    )


def bootstrap_runtime(
    expert_mode: bool = False,
    provider: str | None = None,
    model: str | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> AppRuntime:
    config = load_config(expert_mode=expert_mode, cli_overrides=cli_overrides)
    logs_root = Path(config.runtime.logs_root)
    if not logs_root.is_absolute():
        logs_root = _PROJECT_ROOT / logs_root
    config = replace(
        config,
        runtime=replace(config.runtime, logs_root=str(logs_root)),
    )
    # 锚定 externalize_dir 等治理层相对路径到项目根（同 logs_root 处理）
    config = _resolve_relative_paths(config)

    # Thread-level setup — journal created BEFORE MCP/LLM loading.
    thread_id = _make_thread_id()
    threads_root = logs_root / "threads"
    thread_dir = threads_root / thread_id
    thread_dir.mkdir(parents=True, exist_ok=True)
    thread_journal = RunJournal(
        run_id=thread_id,
        events_path=thread_dir / "thread-events.jsonl",
    )
    thread_journal.append("thread.started", {
        "expert_mode": expert_mode,
        "provider": provider or "default",
    })

    # LLM construction — 角色化智能路由（deepseek 兜底），或 CLI --provider 强制单 provider。
    from agent4ml.backend.agents.config.model_router import ModelRouter

    router = ModelRouter()
    if provider:
        researcher_model = router.build_single(provider, model)
        reporter_model = researcher_model
        thread_journal.append("llm.constructed", {
            "mode": "single",
            "provider": provider,
            "model": model or "default",
        })
        researcher_model_name = model or provider
    else:
        researcher_model = router.build_model("researcher")
        reporter_model = router.build_model("reporter")
        thread_journal.append("llm.constructed", {
            "mode": "routed",
            "researcher_chain": router.chain_names("researcher"),
            "reporter_chain": router.chain_names("reporter"),
        })
        researcher_model_name = "routed:" + ",".join(router.chain_names("researcher"))

    # MCP tool loading — 通过 McpManager 门面加载，配置化 + 熔断器 + fallback。
    tools: dict[str, Any] = {}

    # builtin 工具（ddg_search / read_snapshot）——始终注册，MCP 未启用时的唯一搜索来源。
    # 之前遗漏此调用导致 MCP 关闭时 agent 无任何搜索工具可用。
    builtin_tools = get_available_tools(groups=["core"])
    for t in builtin_tools:
        tools[t.name] = t
    thread_journal.append("builtin.tools_loaded", {"tools": list(tools.keys())})

    mcp_manager = None
    mcp_audit_middleware = None
    if _check_node_available():
        try:
            from agent4ml.backend.agents.mcp import build_mcp_manager

            mcp_manager = build_mcp_manager()
            if mcp_manager is not None:
                import asyncio

                try:
                    asyncio.get_running_loop()
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        pool.submit(asyncio.run, mcp_manager.load_startup()).result()
                except RuntimeError:
                    asyncio.run(mcp_manager.load_startup())
                mcp_tools = mcp_manager.get_tools(["core", "deferred"])
                for tool in mcp_tools:
                    tools[tool.name] = tool
                search_tool = select_search_tool(mcp_tools)
                if search_tool:
                    tools["web_search_mcp"] = search_tool
                mcp_audit_middleware = mcp_manager.get_audit_middleware()
                # 注入 tool_metadata 到 context_governance.params，供外化层按工具调阈值
                tool_metadata = mcp_manager.registry.get_all_metadata()
                if tool_metadata:
                    cg_params = dict(config.context_governance.params)
                    cg_params["tool_metadata"] = tool_metadata
                    config = replace(
                        config,
                        context_governance=replace(config.context_governance, params=cg_params),
                    )
                thread_journal.append("mcp.loaded", {
                    "tools": list(tools.keys()),
                    "count": len(tools),
                })
            else:
                thread_journal.append("mcp.skipped", {"reason": "disabled or no servers"})
        except Exception as exc:
            thread_journal.append("mcp.load_failed", {"error": str(exc)})
    else:
        thread_journal.append("mcp.skipped", {"reason": "npx not found"})
        print(
            "Node.js / npx not found; MCP search disabled.",
            file=sys.stderr,
        )

    # Registry + LeaderAgent — built ONCE per thread, reused across runs.
    # Sandbox 装配（Grill #9：config 配了 provider 就加载，不论模式）
    sandbox_provider = _load_sandbox_provider(config)
    sandbox_tools = []
    artifact_server = None
    if sandbox_provider is not None:
        from agent4ml.backend.agents.artifacts.server import ArtifactServer
        from agent4ml.backend.agents.sandbox.integration.tools import make_sandbox_tools
        from agent4ml.backend.agents.sandbox.integration.bootstrap_sandbox import (
            register_sandbox_shutdown,
        )

        sandbox_tools = make_sandbox_tools(
            sandbox_provider,
            allow_host_bash=getattr(config.sandbox, "allow_host_bash", True),
        )
        register_sandbox_shutdown(sandbox_provider)
        artifact_server = ArtifactServer()
        artifact_server.start()

    # Skill 模块加载 — build_skill_manager 读 .env，enabled=false 或无目录返 None。
    skill_manager = None
    skill_injection_middleware = None
    skill_metrics_middleware = None
    try:
        from agent4ml.backend.agents.skill import build_skill_manager

        skill_manager = build_skill_manager()
        if skill_manager is not None:
            skill_manager.load_startup(llm=researcher_model)
            skill_injection_middleware = skill_manager.get_injection_middleware()
            skill_metrics_middleware = skill_manager.get_metrics_middleware()
            thread_journal.append("skill.loaded", {
                "skills": [s["name"] for s in skill_manager.list_skills()],
            })
            # 自进化装配（Layer 2a）：evolve_enabled=true 时建 EvolutionManager 注入
            if skill_manager.config.evolve_enabled:
                try:
                    skill_manager.set_evolution_manager(
                        _build_evolution_manager(skill_manager, researcher_model, thread_journal)
                    )
                    thread_journal.append("skill.evolve_loaded", {})
                except Exception as exc:
                    thread_journal.append("skill.evolve_load_failed", {"error": str(exc)})
                # L3 eval 装配：eval_config.enabled=true 时建 EvalLayer 注入
                if skill_manager.config.eval_config.enabled:
                    try:
                        skill_manager.set_eval_layer(
                            _build_eval_layer(skill_manager, researcher_model)
                        )
                        thread_journal.append("skill.eval_loaded", {})
                    except Exception as exc:
                        thread_journal.append("skill.eval_load_failed", {"error": str(exc)})
        else:
            thread_journal.append("skill.skipped", {"reason": "disabled or no skills dir"})
    except Exception as exc:
        thread_journal.append("skill.load_failed", {"error": str(exc)})

    all_tools = {**tools, **{t.name: t for t in sandbox_tools}}

    # Multi-Agent orchestration 装配 — enabled=false 时返空 setup（lead agent 行为不变）
    ma_config = load_multiagent_config()

    # Bug A 修复：注入 agent_factory 让 SubagentRuntime 可用（设计文档 46 §4.1）
    # _subagent_factory 构造 leaf-role lead agent（复用 lead agent 构造，但 leaf 不能再 delegate）。
    # 闭包捕获本函数局部变量（registry / config / sandbox_provider / 各 middleware）。
    def _subagent_factory() -> Any:
        """Leaf-role lead agent factory for self-copy subagent.

        复用 lead agent 构造逻辑，但 leaf role 限制：
        - specialist_tools=None（leaf 看不到 delegate_to_* tool，不能再 spawn）
        - orchestration_middleware=None（leaf 不挂 OrchestrationMiddleware）
        这些限制让子 agent 无法递归 spawn（INVARIANT：max_spawn_depth=1 leaf-only MVP）。
        """
        return make_lead_agent(
            expert_mode=expert_mode,
            capability_registry=registry,
            context_governance=config.context_governance,
            sandbox_provider=sandbox_provider,
            artifact_server=artifact_server,
            mcp_audit_middleware=mcp_audit_middleware,
            skill_injection_middleware=skill_injection_middleware,
            skill_metrics_middleware=skill_metrics_middleware,
            specialist_tools=None,              # leaf 不能再 delegate
            orchestration_middleware=None,     # leaf 不挂 OrchestrationMiddleware
            memory_provider=memory_provider,
            memory_config=config.memory,
            memory_worker=get_memory_worker(),
        )

    ma_setup = setup_multiagent(
        ma_config,
        agent_factory=_subagent_factory,
    )

    # 同步 AppConfig.memory → 全局单例（middleware/worker/manager 从 get_memory_config() 取）
    set_memory_config(config.memory)
    memory_provider = _load_memory_provider(config)
    memory_worker = None
    if memory_provider is not None:
        memory_worker = start_memory_worker(memory_provider.manager(), researcher_model)
        import atexit
        atexit.register(shutdown_memory_worker)
    registry = CapabilityRegistry(
        models={"researcher": researcher_model, "reporter": reporter_model},
        tools=all_tools,
        reporter=MarkdownReporter(),
        artifact_store=LocalArtifactStore(),
        sandbox_provider=sandbox_provider,
        skill_store=skill_manager.store if skill_manager else None,
        specialist_registry=ma_setup.specialist_registry,
        subagent_provider=ma_setup.subagent_provider,
        memory_provider=memory_provider,
    )
    leader_agent = make_lead_agent(
        expert_mode=expert_mode,
        capability_registry=registry,
        context_governance=config.context_governance,
        sandbox_provider=sandbox_provider,
        artifact_server=artifact_server,
        mcp_audit_middleware=mcp_audit_middleware,
        skill_injection_middleware=skill_injection_middleware,
        skill_metrics_middleware=skill_metrics_middleware,
        specialist_tools=list(ma_setup.specialist_tools) if ma_setup.specialist_tools else None,
        orchestration_middleware=ma_setup.orchestration_middleware,
        memory_provider=memory_provider,
        memory_config=config.memory,
        memory_worker=memory_worker,
    )
    thread_journal.append("agent.constructed", {
        "expert_mode": expert_mode,
        "middleware_count": 6,
        "tools_count": len(tools),
    })

    return AppRuntime(
        config=config,
        capability_registry=registry,
        run_manager=RunManager(config),
        researcher_model_name=researcher_model_name,
        thread_id=thread_id,
        thread_dir=thread_dir,
        thread_journal=thread_journal,
        leader_agent=leader_agent,
        mcp_manager=mcp_manager,
        artifact_server=artifact_server,
        skill_manager=skill_manager,
        multiagent_setup=ma_setup,
    )
