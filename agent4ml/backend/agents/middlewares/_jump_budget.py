"""Shared jump_to model budget for TodoMiddleware Layer 2 + ReflectionMiddleware.

D6: 两个退出闸门共享 per-run 跳转次数预算（合计 ≤3），防双重强制死循环。
按 (thread_id, run_id) key 隔离，模块级 dict + lock。
"""

from __future__ import annotations

import threading

from agent4ml.backend.agents.middlewares.run_journal_middleware import _get_runtime_value
from langgraph.runtime import Runtime

_MAX_TOTAL_JUMPS = 3

_lock = threading.Lock()
_counts: dict[tuple[str, str], int] = {}


def _key(runtime: Runtime) -> tuple[str, str]:
    tid = str(_get_runtime_value(runtime, "thread_id", None) or "default")
    rid = str(_get_runtime_value(runtime, "run_id", None) or "default")
    return (tid, rid)


def try_consume(runtime: Runtime, max_total: int = _MAX_TOTAL_JUMPS) -> bool:
    """原子地检查+消费一次跳转预算。True=已消费（预算可用），False=已耗尽。"""
    k = _key(runtime)
    with _lock:
        n = _counts.get(k, 0)
        if n >= max_total:
            return False
        _counts[k] = n + 1
        return True


def remaining(runtime: Runtime, max_total: int = _MAX_TOTAL_JUMPS) -> int:
    k = _key(runtime)
    with _lock:
        return max(0, max_total - _counts.get(k, 0))


def clear(runtime: Runtime) -> None:
    k = _key(runtime)
    with _lock:
        _counts.pop(k, None)
