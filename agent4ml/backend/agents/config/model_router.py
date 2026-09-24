"""ModelRouter — 角色化智能路由，按 MODEL_ROUTES 链构造 FallbackChatModel，deepseek 兜底。

- build_model(role)：按角色路由链构造 FallbackChatModel（多 provider 降级）
- build_single(provider, model)：CLI --provider 强制单 provider（不路由，测试/调试用）
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from agent4ml.backend.agents.config.fallback_model import FallbackChatModel
from agent4ml.backend.agents.config.provider_config import (
    ProviderConfig,
    build_chat_model,
    discover_available_providers,
    route_chain_for,
    select_provider_config,
)


class ModelRouter:
    def __init__(self, providers: list[ProviderConfig] | None = None) -> None:
        self._providers = providers if providers is not None else discover_available_providers()

    def build_model(self, role: str) -> BaseChatModel:
        """按 MODEL_ROUTES[role] 链构造 FallbackChatModel，deepseek 兜尾。"""
        chain = route_chain_for(role, self._providers)
        models = [build_chat_model(p) for p in chain]
        return FallbackChatModel(models=models, provider_names=[p.provider for p in chain])

    def build_single(self, provider: str, model: str | None = None) -> BaseChatModel:
        """CLI --provider 强制单 provider（不路由，测试/调试用）。"""
        cfg = select_provider_config(provider=provider, model=model)
        return build_chat_model(cfg)

    def chain_names(self, role: str) -> list[str]:
        chain = route_chain_for(role, self._providers)
        return [p.provider for p in chain]
