"""Memory bootstrap lifecycle — get/reset/shutdown/set + 反射加载 + journal 注入。

承接 `design_docs/53-memory-l4-middleware-bootstrap.md` §4 Step 2。

lifecycle 4 函数（参照 deer-flow SandboxProvider 模式）：
- get_memory_provider(): 懒加载 + 双检锁 + 反射 config.use
- reset_memory_provider(): 清缓存不关（测试用）
- shutdown_memory_provider(): 关 + 清（hasattr duck-type 委托 store/retriever）
- set_memory_provider(p): 注入（测试用）

_load_memory_provider: 调 build_default_provider(L3 完整版) + journal 注入。
_wrap_store: 装饰器模式，5B 增量索引触发（Batch 3 加）。

INVARIANT：
- 懒加载双检锁：get_memory_provider 线程安全
- shutdown duck-type：hasattr(provider, "shutdown") 委托 store/retriever
- config.use="" 返 None（记忆禁用）
- import 防火墙：bootstrap.py 不 import app
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from agent4ml.backend.agents.memory.config import get_memory_config

logger = logging.getLogger(__name__)

_provider_lock = threading.Lock()
_memory_provider: Any = None


def get_memory_provider() -> Any:
    """懒加载 + 双检锁 + 反射 config.use。None 时返 None（记忆禁用）。

    第一次检查无锁（快速路径），第二次检查有锁（防并发）。
    """
    global _memory_provider
    if _memory_provider is not None:
        return _memory_provider
    with _provider_lock:
        if _memory_provider is not None:
            return _memory_provider
        config = get_memory_config()
        if not config.use:
            return None  # 记忆禁用
        provider = _load_memory_provider(config)
        _memory_provider = provider
        return provider


def reset_memory_provider() -> None:
    """清缓存不关（测试用）。"""
    global _memory_provider
    with _provider_lock:
        _memory_provider = None


def shutdown_memory_provider() -> None:
    """关 + 清（hasattr duck-type 委托 store/retriever）。"""
    global _memory_provider
    with _provider_lock:
        if _memory_provider is not None:
            if hasattr(_memory_provider, "shutdown"):
                _memory_provider.shutdown()
            _memory_provider = None


def set_memory_provider(provider: Any) -> None:
    """注入（测试用）。"""
    global _memory_provider
    with _provider_lock:
        _memory_provider = provider


# ---------------------------------------------------------------------------
# L5 worker lifecycle（块 C3）
# ---------------------------------------------------------------------------
_worker_lock = threading.Lock()
_memory_worker: Any = None


def start_memory_worker(manager: Any, llm: Any) -> Any:
    """启动 memory worker（daemon 线程）。返 worker 实例。

    幂等：已启动则返既有实例。
    LLM 注入：调用方传 llm（避免 worker 反向依赖 app）。
    """
    global _memory_worker
    if _memory_worker is not None:
        return _memory_worker
    from agent4ml.backend.agents.memory.worker import MemoryWorker
    with _worker_lock:
        if _memory_worker is not None:
            return _memory_worker
        worker = MemoryWorker(manager=manager, llm=llm)
        worker.start()
        _memory_worker = worker
        return worker


def shutdown_memory_worker(timeout: float = 5.0) -> None:
    """drain + 关闭 worker。"""
    global _memory_worker
    with _worker_lock:
        if _memory_worker is None:
            return
        _memory_worker.shutdown(timeout=timeout)
        _memory_worker = None


def get_memory_worker() -> Any:
    """取当前 worker 实例（可能为 None）。"""
    with _worker_lock:
        return _memory_worker


def _load_memory_provider(config: Any) -> Any:
    """反射加载 + build_default_provider + journal 注入。

    L3 完整版：build_default_provider(store/retriever 从 config 实例化)。
    journal 注入：_make_journal_callback（RunJournal 可用时 lambda，不可用时 None）。
    _wrap_store（5B 增量索引）在 Batch 3 加。
    """
    from agent4ml.backend.agents.memory.strategies.default.strategy import (
        build_default_provider,
    )

    journal = _make_journal_callback()
    provider = build_default_provider(journal=journal)
    _wrap_store(provider.store(), provider.retriever())  # 5B 增量索引触发
    return provider


def _make_journal_callback() -> Any:
    """Layer 4 注入 RunJournal（L2 manager emit memory.* 事件）。

    RunJournal 可用时返 lambda（调 journal.append(event, payload)）；
    不可用时返 None（不发事件）。
    """
    try:
        from agent4ml.backend.agents.journal import get_run_journal

        journal = get_run_journal()
        if journal is None:
            return None
        return lambda event, payload: journal.append(event, payload)
    except Exception as exc:
        logger.debug("RunJournal unavailable, memory journal events disabled: %s", exc)
        return None


def _wrap_store(store: Any, retriever: Any) -> None:
    """装饰器模式：包装 store.add/update/batch_update/remove 后调 retriever.on_trace_*。

    5B 增量索引触发（50 §6.5）：store 变更后 retriever 索引同步更新。
    运行时方法替换，不改 store 类。
    """
    original_add = store.add
    original_update = store.update
    original_batch_update = store.batch_update
    original_remove = store.remove

    def wrapped_add(trace: Any) -> None:
        original_add(trace)
        retriever.on_trace_added(trace)

    def wrapped_update(trace: Any) -> None:
        original_update(trace)
        retriever.on_trace_updated(trace)

    def wrapped_batch_update(traces: list) -> None:
        original_batch_update(traces)
        for t in traces:
            retriever.on_trace_updated(t)

    def wrapped_remove(trace_id: str) -> None:
        original_remove(trace_id)
        retriever.on_trace_removed(trace_id)

    store.add = wrapped_add
    store.update = wrapped_update
    store.batch_update = wrapped_batch_update
    store.remove = wrapped_remove
