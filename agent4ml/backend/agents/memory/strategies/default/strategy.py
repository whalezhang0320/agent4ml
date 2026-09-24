"""默认 MemoryProvider 主入口（组合 store + retriever + manager）。

承接 `Hezao-MemDesign-Docs/agent4ml/48-memory-l1-base-layer.md` §4 Step 6.3 骨架
+ `49-memory-l2-default-strategies.md` §4 Step 5。

Layer 1：空骨架，build_default_provider() 返 NotImplementedError 占位。
Layer 2：部分组装 — 接收 store + retriever 参数注入，组装 manager + decay + forget。
Layer 3（本实现）：完整版 — 从 config 实例化 MarkdownFileStore + HybridRetriever，组装 DefaultMemoryProvider。

INVARIANT：
- DefaultMemoryProvider frozen dataclass，三组件构造时注入，运行时不可变
- shutdown duck-type：委托 store/retriever（hasattr 检查）
- store/retriever/journal 由调用方注入（Layer 2 测试 mock，Layer 3 后真实实例）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from agent4ml.backend.agents.memory.config import get_memory_config
from agent4ml.backend.agents.memory.memory_manager import MemoryManager
from agent4ml.backend.agents.memory.memory_provider import MemoryProvider
from agent4ml.backend.agents.memory.memory_store import MemoryStore
from agent4ml.backend.agents.memory.retriever import Retriever
from agent4ml.backend.agents.memory.strategies.default.decay import EbbinghausDecayPolicy
from agent4ml.backend.agents.memory.strategies.default.forget import CompositeForgetPolicy
from agent4ml.backend.agents.memory.strategies.default.manager import DefaultMemoryManager
from agent4ml.backend.agents.memory.strategies.default.retriever import HybridRetriever
from agent4ml.backend.agents.memory.strategies.default.store import MarkdownFileStore


@dataclass(frozen=True)
class DefaultMemoryProvider:
    """默认 MemoryProvider 实现（组合 store + retriever + manager）。

    frozen dataclass，三组件构造时注入，运行时不可变。
    Lifecycle：构造即就绪，shutdown 走 hasattr duck-type（委托给 store/retriever）。
    """

    _store: MemoryStore
    _retriever: Retriever
    _manager: MemoryManager

    def store(self) -> MemoryStore:
        return self._store

    def retriever(self) -> Retriever:
        return self._retriever

    def manager(self) -> MemoryManager:
        return self._manager

    def shutdown(self) -> None:
        """可选 shutdown：委托给 store / retriever（若有）。

        hasattr duck-type：MarkdownFileStore 无 shutdown（纯文件），SQLiteShadowStore 有。
        """
        if hasattr(self._store, "shutdown"):
            self._store.shutdown()
        if hasattr(self._retriever, "shutdown"):
            self._retriever.shutdown()


def build_default_provider(
    *,
    store: MemoryStore | None = None,
    retriever: Retriever | None = None,
    decay_policy: EbbinghausDecayPolicy | None = None,
    forget_policy: CompositeForgetPolicy | None = None,
    journal: Callable[[str, dict], None] | None = None,
) -> MemoryProvider:
    """组装默认 MemoryProvider（Layer 3 完整版）。

    Args:
        store: 持久化后端（None 时从 config 实例化 MarkdownFileStore）
        retriever: 检索后端（None 时从 store + decay_policy 实例化 HybridRetriever）
        decay_policy: 衰减策略（None 时默认 EbbinghausDecayPolicy）
        forget_policy: 遗忘策略（None 时默认 CompositeForgetPolicy）
        journal: 事件回调（traceability B，None 时不发事件；Layer 4 注入 RunJournal）

    Returns:
        DefaultMemoryProvider（组合三组件）

    Layer 3：store/retriever 默认从 config 实例化（Layer 4 bootstrap 调此函数）。
    Layer 4 注入 journal（RunJournal）。
    """
    config = get_memory_config()
    decay = decay_policy or EbbinghausDecayPolicy()
    forget = forget_policy or CompositeForgetPolicy(decay)

    # Layer 3 完整实例化（store/retriever 未注入时从 config 建）
    if store is None:
        store = MarkdownFileStore(config.storage_path)
    if retriever is None:
        retriever = HybridRetriever(store, decay)

    manager = DefaultMemoryManager(
        store=store,
        decay_policy=decay,
        forget_policy=forget,
        journal=journal,  # traceability B 透传
    )
    return DefaultMemoryProvider(
        _store=store,
        _retriever=retriever,
        _manager=manager,
    )
