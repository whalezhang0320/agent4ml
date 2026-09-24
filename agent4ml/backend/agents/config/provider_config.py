from __future__ import annotations

import os
from dataclasses import dataclass

from agent4ml.backend.agents.config.provider_profile import (
    PROVIDER_PROFILES,
    ProviderProfile,
    get_provider_profile,
)


class ProviderConfigError(ValueError):
    """Raised when model provider config is missing or invalid."""


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    model: str
    api_key: str
    base_url: str | None
    priority: int
    default: bool
    enabled: bool
    window: int = 0  # 上下文窗口（token），0=未知，由 resolve_window_size 兜底

    def require_api_key(self) -> str:
        if not self.api_key:
            raise ProviderConfigError(f"api_key is empty for provider: {self.provider}")
        return self.api_key


def _resolve_profile(profile: ProviderProfile) -> ProviderConfig:
    """从 ProviderProfile 读 env 解析为 ProviderConfig。

    env 变量名在 profile 声明，值在此处读取（不在模块加载时读，利于测试 monkeypatch）。
    每个 provider 支持 {NAME}_ENABLED=false 单独禁用。
    """
    api_key = os.environ.get(profile.env_key, "")
    base_url = os.environ.get(profile.env_base_url, "") or profile.default_base_url
    model = os.environ.get(profile.env_model, "") or profile.default_model
    enabled_env = os.environ.get(f"{profile.name.upper()}_ENABLED", "true").lower()
    return ProviderConfig(
        provider=profile.name,
        model=model,
        api_key=api_key,
        base_url=base_url if base_url else None,
        priority=profile.priority,
        default=profile.is_default,
        enabled=enabled_env != "false",
        window=profile.default_window,
    )


def _all_configs() -> list[ProviderConfig]:
    """解析全部 provider profile 为 ProviderConfig（含 disabled）。"""
    return [_resolve_profile(p) for p in PROVIDER_PROFILES]


def select_provider_config(
    provider: str | None = None,
    model: str | None = None,
) -> ProviderConfig:
    candidates = [c for c in _all_configs() if c.enabled]
    if provider:
        selected = _find_provider(candidates, provider)
    else:
        defaults = [c for c in candidates if c.default]
        selected = sorted(defaults or candidates, key=lambda item: item.priority)[0]
    if model:
        return ProviderConfig(
            provider=selected.provider,
            model=model,
            api_key=selected.api_key,
            base_url=selected.base_url,
            priority=selected.priority,
            default=selected.default,
            enabled=selected.enabled,
            window=selected.window,
        )
    return selected


def get_provider_config(provider: str) -> ProviderConfig:
    return select_provider_config(provider=provider)


def _find_provider(candidates: list[ProviderConfig], provider: str) -> ProviderConfig:
    for candidate in candidates:
        if candidate.provider == provider:
            return candidate
    raise ProviderConfigError(f"provider not configured: {provider}")


# ---------------------------------------------------------------------------
# 智能路由（deepseek 兜底）
# ---------------------------------------------------------------------------

# 角色路由链：按顺序偏好，链尾恒含 deepseek（兜底）。
MODEL_ROUTES: dict[str, list[str]] = {
    "researcher": ["openai", "qwen", "anthropic", "gemini", "deepseek"],
    "reporter": ["qwen", "deepseek"],
    "reflection": ["deepseek"],
}


def discover_available_providers() -> list[ProviderConfig]:
    """返回 enabled 且 api_key 非空的 provider（no_key_required 的除外）。按 priority 升序。"""
    available = [
        c for c in _all_configs()
        if c.enabled and (c.api_key or _is_no_key(c.provider))
    ]
    return sorted(available, key=lambda p: p.priority)


def _is_no_key(provider: str) -> bool:
    """provider 是否无需 API key（fake / ollama）。"""
    profile = get_provider_profile(provider)
    return profile is not None and profile.no_key_required


def route_chain_for(role: str, providers: list[ProviderConfig]) -> list[ProviderConfig]:
    """按 MODEL_ROUTES[role] 顺序筛 provider；保证 deepseek 在链尾（若可用）。

    - role 未配置 → 默认 ["deepseek"]
    - 链空或尾非 deepseek 且 deepseek 可用 → 追加 deepseek 兜底
    - 无任何可用 → 抛 ProviderConfigError
    """
    route = MODEL_ROUTES.get(role, ["deepseek"])
    by_name = {p.provider: p for p in providers}
    chain = [by_name[name] for name in route if name in by_name]
    if "deepseek" in by_name and (not chain or chain[-1].provider != "deepseek"):
        chain.append(by_name["deepseek"])
    if not chain:
        raise ProviderConfigError(f"no available provider for role: {role}")
    return chain


def build_chat_model(config: ProviderConfig):
    """根据 ProviderConfig 构造 BaseChatModel。按 provider kind 分发。

    optional provider（anthropic/gemini/ollama）的 langchain 包未安装时
    抛 ProviderConfigError 提示安装对应 optional-dependencies。
    """
    profile = get_provider_profile(config.provider)
    if profile is None:
        raise ProviderConfigError(f"unsupported provider: {config.provider}")
    if not profile.no_key_required:
        config.require_api_key()

    kind = profile.kind
    if kind == "deepseek":
        from langchain_deepseek import ChatDeepSeek
        kwargs: dict = {"model": config.model, "api_key": config.api_key}
        if config.base_url:
            kwargs["api_base"] = config.base_url
        return ChatDeepSeek(**kwargs)

    if kind == "openai_compat":
        from langchain_openai import ChatOpenAI
        kwargs = {"model": config.model, "api_key": config.api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOpenAI(**kwargs)

    if kind == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ProviderConfigError(
                "langchain-anthropic not installed. Run: pip install langchain-anthropic"
            )
        kwargs = {"model": config.model, "api_key": config.api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatAnthropic(**kwargs)

    if kind == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ProviderConfigError(
                "langchain-google-genai not installed. Run: pip install langchain-google-genai"
            )
        kwargs = {"model": config.model, "google_api_key": config.api_key}
        return ChatGoogleGenerativeAI(**kwargs)

    if kind == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ProviderConfigError(
                "langchain-ollama not installed. Run: pip install langchain-ollama"
            )
        kwargs = {"model": config.model}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOllama(**kwargs)

    if kind == "fake":
        from langchain_core.language_models.fake_chat_models import FakeListChatModel
        return FakeListChatModel(responses=["fake response from fake provider"])

    raise ProviderConfigError(f"unsupported provider kind: {kind}")
