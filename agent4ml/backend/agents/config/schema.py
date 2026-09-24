from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agent4ml.backend.agents.memory.config import MemoryConfig
from agent4ml.backend.agents.sandbox.integration.config import SandboxConfig
from agent4ml.backend.agents.skill.config import SkillConfig


@dataclass(frozen=True)
class RuntimeConfig:
    expert_mode: bool = False
    timezone: str = "Asia/Shanghai"
    max_loop_steps: int = 4
    timeout_seconds: int = 120
    output_root: str = ".agent4ml"
    logs_root: str = ".agent4ml/logs"
    plan_enabled: bool = True
    reflection_enabled: bool = False
    graph_node_multiplier: int = 200


@dataclass(frozen=True)
class HitlConfig:
    """Human-in-the-loop configuration for long-task stall detection and help."""
    capability_failure_threshold: int = 2
    error_pattern_threshold: int = 3
    todo_stagnation_rounds: int = 5
    no_progress_timeout: int = 180
    max_help_requests: int = 3
    activity_heartbeat_interval: int = 10
    cancel_grace_period: int = 5
    steer_enabled: bool = True


@dataclass(frozen=True)
class ContextGovernanceConfig:
    """上下文治理层配置（策略层）。公共层 middleware 固定挂，不经此配置。"""

    strategy: str = "default"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelConfig:
    researcher_model: str
    reporter_model: str


@dataclass(frozen=True)
class ToolConfig:
    web_search_mcp: str = "fake"
    tool_search_default: bool = True


@dataclass(frozen=True)
class MiddlewareConfig:
    enabled: tuple[str, ...] = field(default_factory=tuple)
    summarization: bool = False
    todo: bool = False
    title: bool = False


@dataclass(frozen=True)
class ReportingConfig:
    save_artifact: bool = True
    artifact_format: str = "markdown"


@dataclass(frozen=True)
class ObservabilityConfig:
    event_log_enabled: bool = True
    log_level: str = "INFO"


@dataclass(frozen=True)
class AppConfig:
    name: str
    environment: str
    runtime: RuntimeConfig
    models: ModelConfig
    tools: ToolConfig
    middleware: MiddlewareConfig
    reporting: ReportingConfig
    observability: ObservabilityConfig
    context_governance: ContextGovernanceConfig = field(default_factory=ContextGovernanceConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    skill: SkillConfig = field(default_factory=SkillConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
