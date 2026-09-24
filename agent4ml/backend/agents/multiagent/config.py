"""MultiAgentConfig — multi-agent orchestration configuration.

设计（spec.md bootstrap Requirement + design.md §2）:
- frozen dataclass，enabled=false 默认（opt-in）
- STARTUP_ONLY_FIELDS 标记启动时确定的字段（不可热切换）
- 从 env vars 加载（AGENT4ML_MULTIAGENT_*）
- OrchestrationState + merge_orchestration 在 state/types.py + state/reducers.py（Batch 3 已实现）
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class L2Config:
    """L2 evolution layer config (R4 + R6 + R7).

    enabled=false default (data-driven trigger).
    """

    enabled: bool = False
    cron_interval_hours: float = 6.0
    cooldown_seconds: float = 3600.0
    anti_loop_window: int = 5
    failure_window_hours: float = 24.0
    failure_threshold: int = 5
    degradation_min_invoked: int = 5
    degradation_threshold: float = 0.4
    cost_alert_usd: float = 1.0
    latency_alert_seconds: float = 300.0
    blocked_auto_release_hours: float = 24.0
    evolution_model: str | None = None
    eval_timeout_seconds: float = 1800.0
    eval_sample_min: int = 10
    eval_sample_max: int = 15
    eval_task_max_reuse: int = 3
    intent_llm_enabled: bool = False
    intent_model: str | None = None
    intent_confidence_threshold: float = 0.7
    intent_delegate_rate_threshold: float = 0.2
    intent_ability_failure_threshold: float = 0.5
    intent_metadata_sample_size: int = 20
    l3_enabled: bool = False


@dataclass(frozen=True)
class L3Config:
    """L3 eval layer config (43 doc §8).

    enabled=false default (data-driven trigger).
    llm_judge_weights 复用 skill TaskQualityJudge 权重值（D-L3-13）.
    """

    enabled: bool = False
    default_eval_method: str = "programmatic"
    llm_judge_model: str | None = None
    llm_judge_weights: dict = field(
        default_factory=lambda: {
            "task_completion": 0.50,
            "response_quality": 0.35,
            "efficiency": 0.05,
            "tool_usage": 0.10,
        }
    )
    health_check_window: int = 20
    degradation_threshold: float = 0.4
    degradation_delta: float = 0.15
    decision_log_retention_days: int = 90
    decision_log_archive_enabled: bool = True


@dataclass(frozen=True)
class SpecialistBudgetLimit:
    """Per-specialist daily budget limit (R5.1)."""

    per_day_tokens: int = 200000
    per_day_cost_usd: float = 20.0
    per_day_calls: int = 50


@dataclass(frozen=True)
class BudgetConfig:
    """Budget config (R5). warning_threshold=0.8 (80% warning)."""

    codex: SpecialistBudgetLimit = field(default_factory=SpecialistBudgetLimit)
    claude: SpecialistBudgetLimit = field(default_factory=SpecialistBudgetLimit)
    subagent: SpecialistBudgetLimit = field(default_factory=SpecialistBudgetLimit)
    pi: SpecialistBudgetLimit = field(default_factory=SpecialistBudgetLimit)
    warning_threshold: float = 0.8


@dataclass(frozen=True)
class MultiAgentConfig:
    """Multi-agent orchestration configuration.

    enabled=true 默认——default + expert 模式都装配 multiagent。
    用 AGENT4ML_MULTIAGENT_ENABLED=false 可显式关闭。
    """

    enabled: bool = True
    specialists_use: tuple[str, ...] = ("pi", "codex", "claude", "subagent")
    auto_approve: bool = True
    max_concurrent: int = 1
    timeout_seconds: int = 600
    max_steps: int = 50
    subagent_tool_groups: tuple[str, ...] = ("core",)
    subagent_max_steps: int = 20
    subagent_timeout_seconds: int = 300
    metrics_db_path: str = ".agent4ml/multiagent.db"
    metrics_health_threshold: float = 0.4
    metrics_min_invoked: int = 5
    # Pi specialist 配置（决策 2 + 决策 3 + 决策 5）
    specialists_pi_provider: str = ""
    specialists_pi_api_key: str = ""
    specialists_pi_auto_install: bool = True
    specialists_pi_model: str = ""
    specialists_pi_thinking_level: str = "medium"
    # L2 evolution layer config (default enabled=false, data-driven trigger)
    l2: L2Config = field(default_factory=L2Config)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    # L3 eval layer config (default enabled=false, data-driven trigger)
    l3: L3Config = field(default_factory=L3Config)


STARTUP_ONLY_FIELDS = frozenset({
    "enabled",
    "specialists_use",
    "metrics_db_path",
    "l2.enabled",
    "l3.enabled",
})


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in ("true", "1", "yes")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_tuple(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    val = os.getenv(name, "")
    if not val:
        return default
    return tuple(s.strip() for s in val.split(",") if s.strip())


def load_multiagent_config() -> MultiAgentConfig:
    """Load multiagent config from env vars (AGENT4ML_MULTIAGENT_*)."""
    return MultiAgentConfig(
        enabled=_env_bool("AGENT4ML_MULTIAGENT_ENABLED", True),
        specialists_use=_env_tuple("AGENT4ML_MULTIAGENT_SPECIALISTS", ("pi", "codex", "claude", "subagent")),
        auto_approve=_env_bool("AGENT4ML_MULTIAGENT_AUTO_APPROVE", True),
        max_concurrent=_env_int("AGENT4ML_MULTIAGENT_MAX_CONCURRENT", 1),
        timeout_seconds=_env_int("AGENT4ML_MULTIAGENT_TIMEOUT", 600),
        max_steps=_env_int("AGENT4ML_MULTIAGENT_MAX_STEPS", 50),
        subagent_tool_groups=_env_tuple("AGENT4ML_MULTIAGENT_SUBAGENT_TOOL_GROUPS", ("core",)),
        subagent_max_steps=_env_int("AGENT4ML_MULTIAGENT_SUBAGENT_MAX_STEPS", 20),
        subagent_timeout_seconds=_env_int("AGENT4ML_MULTIAGENT_SUBAGENT_TIMEOUT", 300),
        metrics_db_path=os.getenv("AGENT4ML_MULTIAGENT_DB_PATH", ".agent4ml/multiagent.db"),
        metrics_health_threshold=_env_float("AGENT4ML_MULTIAGENT_HEALTH_THRESHOLD", 0.4),
        metrics_min_invoked=_env_int("AGENT4ML_MULTIAGENT_MIN_INVOKED", 5),
        # Pi specialist 配置（决策 2 + 决策 3 + 决策 5）
        specialists_pi_provider=os.getenv("AGENT4ML_MULTIAGENT_PI_PROVIDER", ""),
        specialists_pi_api_key=os.getenv("AGENT4ML_MULTIAGENT_PI_API_KEY", ""),
        specialists_pi_auto_install=_env_bool("AGENT4ML_MULTIAGENT_PI_AUTO_INSTALL", True),
        specialists_pi_model=os.getenv("AGENT4ML_MULTIAGENT_PI_MODEL", ""),
        specialists_pi_thinking_level=os.getenv("AGENT4ML_MULTIAGENT_PI_THINKING", "medium"),
        l2=L2Config(
            enabled=_env_bool("AGENT4ML_MULTIAGENT_L2_ENABLED", False),
            cron_interval_hours=_env_float("AGENT4ML_MULTIAGENT_L2_CRON_HOURS", 6.0),
            cooldown_seconds=_env_float("AGENT4ML_MULTIAGENT_L2_COOLDOWN", 3600.0),
            evolution_model=os.getenv("AGENT4ML_MULTIAGENT_L2_MODEL") or None,
            l3_enabled=_env_bool("AGENT4ML_MULTIAGENT_L3_ENABLED", False),
        ),
        l3=L3Config(
            enabled=_env_bool("AGENT4ML_MULTIAGENT_L3_ENABLED", False),
            llm_judge_model=os.getenv("AGENT4ML_MULTIAGENT_L3_JUDGE_MODEL") or None,
        ),
    )
