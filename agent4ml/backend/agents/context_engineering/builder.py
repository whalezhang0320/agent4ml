"""治理 middleware 装配构建器。

組裝公共 2 固定（TaggedContext/MessageNormalizer）+ 1 StrategyMiddleware
（持 config 選定的策略 bundle）。策略 bundle 未註冊時跳過 StrategyMiddleware + 警告
（過渡期 minimal 等非 bundle 名只掛公共2，不崩）。

掛載順序：公共 TaggedContext → 公共 MessageNormalizer →
StrategyMiddleware（策略 bundle 6 hook）。
DateInjector 邏輯已併入 TaggedContext（<date> 標籤渲染）。
token 預算 + 硬停由策略 bundle 內部負責（DefaultStrategy 的 BudgetTrackerExecutor
算 fraction + P5/P99 分段），公共層不再重複實現硬停。
"""

from __future__ import annotations

import logging
from typing import Any

from agent4ml.backend.agents.config.schema import ContextGovernanceConfig
from agent4ml.backend.agents.context_engineering import strategies  # noqa: F401  # 触发 bundle 注册
from agent4ml.backend.agents.context_engineering.registry import get_strategy_class
from agent4ml.backend.agents.context_engineering.strategy_middleware import StrategyMiddleware
from agent4ml.backend.agents.middlewares.message_normalizer_middleware import (
    MessageNormalizerMiddleware,
)
from agent4ml.backend.agents.middlewares.tagged_context_middleware import (
    TaggedContextMiddleware,
)

logger = logging.getLogger(__name__)


def build_governance_middlewares(config: ContextGovernanceConfig, model: Any = None, summarize_model: Any = None) -> list:
    """組裝公共 2 固定 + 1 StrategyMiddleware（按 config.strategy 選 bundle，未註冊跳過）。"""
    middlewares: list = [
        TaggedContextMiddleware(),
        MessageNormalizerMiddleware(),
    ]
    try:
        bundle_cls = get_strategy_class(config.strategy)
    except KeyError:
        logger.warning(
            "strategy bundle '%s' not registered, skip StrategyMiddleware (public 2 only)",
            config.strategy,
        )
        return middlewares
    bundle = bundle_cls(config.params, model=model, summarize_model=summarize_model)
    middlewares.append(StrategyMiddleware(bundle, config.params))
    return middlewares
