from __future__ import annotations

from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "name": "agent4ml",
    "environment": "local",
    "runtime": {
        "expert_mode": False,
        "timezone": "Asia/Shanghai",
        "max_loop_steps": 50,
        "timeout_seconds": 120,
        "output_root": ".agent4ml",
        "logs_root": ".agent4ml/logs",
        "plan_enabled": True,
        "reflection_enabled": False,
    },
    "models": {
        "researcher_model": "fake-researcher",
        "reporter_model": "fake-reporter",
    },
    "tools": {
        "web_search_mcp": "fake",
        "tool_search_default": True,
    },
    "middleware": {
        "enabled": (),
        "summarization": False,
        "todo": False,
        "title": False,
    },
    "reporting": {
        "save_artifact": True,
        "artifact_format": "markdown",
    },
    "observability": {
        "event_log_enabled": True,
        "log_level": "INFO",
    },
    "context_governance": {
        "strategy": "default",
        # window 不在此硬编码——由 DefaultStrategy.after_model 每轮调用
        # resolve_window_size(model) 动态解析当前活跃 provider 的真实窗口
        # （FallbackChatModel 穿透到内层 ChatDeepSeek/ChatOpenAI 的 model_name，
        # 命中 _MODEL_WINDOW_MAP：deepseek-v4-flash→200k / deepseek-chat→64k / …）。
        # 治理占比 fraction 以此真实窗口为分母，P5 熔断阈值才准确。
        "params": {},
    },
}

# expert 模式叠加层：load_config(expert_mode=True) 时 deep_merge 到 DEFAULT_CONFIG。
# 替代原 fast/general/expert 三 profile（mode 枚举废弃，改 expert_mode: bool 参数化）。
EXPERT_PROFILE: dict[str, Any] = {
    "runtime": {
        "expert_mode": True,
        "max_loop_steps": 100,
        "plan_enabled": True,
        "reflection_enabled": True,
    },
    "tools": {"tool_search_default": True},
    "middleware": {"enabled": ("todo", "summarization"), "todo": True, "summarization": True},
    "reporting": {"save_artifact": True},
    # 不覆盖 strategy——沿用 DEFAULT_CONFIG 的 "default"（唯一已注册的策略 bundle）。
    # 曾误写 "minimal"（未注册），导致 build_governance_middlewares 静默跳过
    # StrategyMiddleware：budget/fraction 永久停在 0、Compact 进度条不再更新（见 D12）。
}
