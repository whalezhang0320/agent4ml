"""策略 bundle 注册表。

维护 ``name -> bundle_cls`` 映射。``@register_strategy(name)`` 装饰注册，
``get_strategy_class(name)`` 查表实例化。新增策略 bundle 仅需新文件加装饰
并 import 触发注册。
"""

from __future__ import annotations

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)

_STRATEGY_BUNDLES: dict[str, type] = {}


def register_strategy(name: str) -> Callable[[type], type]:
    """装饰器：注册策略 bundle 类到全局表。"""

    def decorator(cls: type) -> type:
        if name in _STRATEGY_BUNDLES and _STRATEGY_BUNDLES[name] is not cls:
            logging.getLogger(__name__).warning(
                "strategy bundle '%s' re-registered: %s -> %s",
                name,
                _STRATEGY_BUNDLES[name].__name__,
                cls.__name__,
            )
        _STRATEGY_BUNDLES[name] = cls
        return cls

    return decorator


def get_strategy_class(name: str) -> type:
    """按名查策略 bundle 类，未注册抛 KeyError。"""
    if name not in _STRATEGY_BUNDLES:
        raise KeyError(
            f"strategy bundle '{name}' not registered. "
            f"Available: {sorted(_STRATEGY_BUNDLES)}"
        )
    return _STRATEGY_BUNDLES[name]


def list_strategies() -> list[str]:
    """列出所有已注册策略 bundle 名。"""
    return sorted(_STRATEGY_BUNDLES)


def clear_strategies() -> None:
    """清空策略 bundle 注册表，用于测试隔离。"""
    _STRATEGY_BUNDLES.clear()
