from __future__ import annotations

import atexit

from agent4ml.backend.agents.sandbox.contracts import SandboxProvider

_registered_providers: list[SandboxProvider] = []
_atexit_registered: bool = False


def register_sandbox_shutdown(provider: SandboxProvider) -> None:
    """注册 provider shutdown 到 atexit（不重复注册信号，Grill #6）。

    多次调用只注册一次 atexit handler，provider 累积到 _registered_providers。
    """
    global _atexit_registered
    _registered_providers.append(provider)
    if not _atexit_registered:
        atexit.register(_shutdown_all_providers)
        _atexit_registered = True


def _shutdown_all_providers() -> None:
    """进程退出时调所有 registered provider 的 shutdown()。"""
    for provider in _registered_providers:
        try:
            provider.shutdown()
        except Exception:
            pass
    _registered_providers.clear()


def _reset_for_testing() -> None:
    """测试隔离：清空 provider 列表 + 重置 atexit 标记。"""
    global _atexit_registered
    _registered_providers.clear()
    _atexit_registered = False
