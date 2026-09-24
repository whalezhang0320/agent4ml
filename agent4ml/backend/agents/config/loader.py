from __future__ import annotations

import os
from copy import deepcopy
from typing import Any

from agent4ml.backend.agents.config.defaults import DEFAULT_CONFIG, EXPERT_PROFILE
from agent4ml.backend.agents.config.schema import (
    AppConfig,
    ContextGovernanceConfig,
    MiddlewareConfig,
    ModelConfig,
    ObservabilityConfig,
    ReportingConfig,
    RuntimeConfig,
    ToolConfig,
)
from agent4ml.backend.agents.memory.config import MemoryConfig
from agent4ml.backend.agents.sandbox.integration.config import SandboxConfig


class ConfigError(ValueError):
    """Raised when config cannot be loaded or validated."""


def load_config(
    expert_mode: bool = False,
    cli_overrides: dict[str, Any] | None = None,
) -> AppConfig:
    overrides = cli_overrides or {}
    # expert_mode: cli_overrides 优先，其次参数，最后默认 False
    selected_expert = bool(overrides.get("expert_mode", expert_mode))

    raw = deepcopy(DEFAULT_CONFIG)
    if selected_expert:
        _deep_merge(raw, EXPERT_PROFILE)
    _apply_cli_overrides(raw, overrides)
    _validate(raw)
    return _build_config(raw)


def _apply_cli_overrides(raw: dict[str, Any], overrides: dict[str, Any]) -> None:
    if "expert_mode" in overrides:
        em = overrides["expert_mode"]
        if not isinstance(em, bool):
            raise ConfigError("expert_mode must be a boolean")
        raw["runtime"]["expert_mode"] = em
        if em is True:
            _deep_merge(raw, EXPERT_PROFILE)
        # False 时保持 DEFAULT（不 merge EXPERT_PROFILE）

    flat_targets = {
        "logs_root": ("runtime", "logs_root"),
        "output_root": ("runtime", "output_root"),
        "researcher_model": ("models", "researcher_model"),
        "reporter_model": ("models", "reporter_model"),
        "save_artifact": ("reporting", "save_artifact"),
    }
    for key, path in flat_targets.items():
        if key not in overrides:
            continue
        section, field = path
        raw[section][field] = overrides[key]


def _validate(raw: dict[str, Any]) -> None:
    models = raw["models"]
    if not models.get("researcher_model"):
        raise ConfigError("researcher_model is required")
    if not models.get("reporter_model"):
        raise ConfigError("reporter_model is required")
    if not isinstance(raw["runtime"].get("expert_mode"), bool):
        raise ConfigError("expert_mode must be a boolean")
    if raw["runtime"]["max_loop_steps"] < 1:
        raise ConfigError("max_loop_steps must be greater than zero")
    if not raw["runtime"].get("logs_root"):
        raise ConfigError("logs_root is required")


def _build_sandbox_config() -> SandboxConfig:
    """从 AGENT4ML_SANDBOX_* 环境变量构造 SandboxConfig（懒加载，use 为空=禁用）。"""
    return SandboxConfig(
        use=os.environ.get("AGENT4ML_SANDBOX_USE", ""),
        allow_host_bash=os.environ.get("AGENT4ML_SANDBOX_ALLOW_HOST_BASH", "true").lower() != "false",
        image=os.environ.get("AGENT4ML_SANDBOX_IMAGE", "all-in-one-sandbox:latest"),
        port=int(os.environ.get("AGENT4ML_SANDBOX_PORT", "18000") or "18000"),
        container_prefix=os.environ.get("AGENT4ML_SANDBOX_CONTAINER_PREFIX", "agent4ml-sandbox"),
        executor=os.environ.get("AGENT4ML_SANDBOX_EXECUTOR", "local") or "local",  # type: ignore[arg-type]
        wsl_distro=os.environ.get("AGENT4ML_SANDBOX_WSL_DISTRO") or None,
        wsl_user=os.environ.get("AGENT4ML_SANDBOX_WSL_USER") or None,
        idle_timeout=int(os.environ.get("AGENT4ML_SANDBOX_IDLE_TIMEOUT", "600") or "600"),
        replicas=int(os.environ.get("AGENT4ML_SANDBOX_REPLICAS", "3") or "3"),
    )


def _build_memory_config() -> MemoryConfig:
    """从 AGENT4ML_MEMORY_* 环境变量构造 MemoryConfig（懒加载，use 为空=禁用）。"""
    return MemoryConfig(
        use=os.environ.get("AGENT4ML_MEMORY_USE", ""),
        storage_path=os.environ.get("AGENT4ML_MEMORY_STORAGE_PATH", ".agent4ml/memory"),
        enable_recall=os.environ.get("AGENT4ML_MEMORY_ENABLE_RECALL", "true").lower() != "false",
        enable_extract=os.environ.get("AGENT4ML_MEMORY_ENABLE_EXTRACT", "false").lower() == "true",
        token_budget=int(os.environ.get("AGENT4ML_MEMORY_TOKEN_BUDGET", "2000") or "2000"),
        phase2={
            "enabled": os.environ.get("AGENT4ML_MEMORY_PHASE2_ENABLED", "false").lower() == "true",
            "trigger_every_n_turns": int(os.environ.get("AGENT4ML_MEMORY_PHASE2_TURNS", "10") or "10"),
            "trigger_on_session_end": True,
        },
    )


def _build_config(raw: dict[str, Any]) -> AppConfig:
    return AppConfig(
        name=raw["name"],
        environment=raw["environment"],
        runtime=RuntimeConfig(**raw["runtime"]),
        models=ModelConfig(**raw["models"]),
        tools=ToolConfig(**raw["tools"]),
        middleware=MiddlewareConfig(**raw["middleware"]),
        reporting=ReportingConfig(**raw["reporting"]),
        observability=ObservabilityConfig(**raw["observability"]),
        context_governance=ContextGovernanceConfig(**raw.get("context_governance", {})),
        sandbox=_build_sandbox_config(),
        memory=_build_memory_config(),
    )


def _deep_merge(target: dict[str, Any], patch: dict[str, Any]) -> None:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
