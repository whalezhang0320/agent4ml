"""MemoryProvider Protocol — 记忆后端能力协议（组合 store + retriever + manager）。

承接 `Hezao-MemDesign-Docs/agent4ml/00-long-term-memory-foundation.md` §7.3 + §8.2
+ `48-memory-l1-base-layer.md` §4 Step 5.7。

组合模型（参照 sandbox Sandbox 组合 Runtime + Translator + Guard）：
- store(): MemoryStore — 持久化后端，默认 MarkdownFileStore（truth source，总在）
- retriever(): Retriever — 检索后端，默认 HybridRetriever（BM25 + optional Vector/Graph）
- manager(): MemoryManager — 四操作编排（Encode/Associate/Consolidate/Reconsolidate）

Lifecycle（参照 deer-flow SandboxProvider 模式）：
- 构造即就绪：构造器读 config + 初始化内部状态，无强制 initialize() 方法
- shutdown 可选：Protocol 不定义 shutdown 抽象方法，走 hasattr(provider, "shutdown") duck-type
  - MarkdownFileStore 不需要 shutdown（纯文件 IO）
  - SQLiteShadowStore 需要（关连接）
  - VectorStore adapter 需要（unload model）
- 模块级 lifecycle 函数（get/reset/shutdown/set_memory_provider）留 Layer 4 bootstrap.py

默认实现：DefaultMemoryProvider（strategies/default/strategy.py 组装，Layer 2/3）。
可替换：VectorMemoryProvider / GraphMemoryProvider（未来，Layer 6）。

INVARIANT:
- 组合模式（换后端只换组件，编排逻辑复用）
- lifecycle duck-type（构造即就绪，无强制 initialize()；shutdown hasattr 检查）
- Protocol 纯契约零实现
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agent4ml.backend.agents.memory.memory_manager import MemoryManager
from agent4ml.backend.agents.memory.memory_store import MemoryStore
from agent4ml.backend.agents.memory.retriever import Retriever


@runtime_checkable
class MemoryProvider(Protocol):
    """记忆后端能力协议（00 §7.3 + §8.2 叠加模式）。

    组合模型（参照 deer-flow SandboxProvider + deer-flow MemoryStorage）：
    - store(): MemoryStore — 持久化后端，默认 MarkdownFileStore（truth source，总在）
    - retriever(): Retriever — 检索后端，默认 HybridRetriever（BM25 + optional Vector/Graph）
    - manager(): MemoryManager — 四操作编排（Encode/Associate/Consolidate/Reconsolidate）

    Lifecycle（参照 deer-flow SandboxProvider 模式）：
    - 构造即就绪：构造器读 config + 初始化内部状态，无强制 initialize() 方法
    - shutdown 可选：Protocol 不定义 shutdown 抽象方法，走 hasattr(provider, "shutdown") duck-type
      MarkdownFileStore 不需要 shutdown（纯文件 IO）
      SQLiteShadowStore 需要（关连接）
      VectorStore adapter 需要（unload model）
    - 模块级 lifecycle 函数（get/reset/shutdown/set_memory_provider）留 Layer 4 bootstrap.py

    默认实现 DefaultMemoryProvider（strategies/default/strategy.py 组装，Layer 2/3）。
    可替换 VectorMemoryProvider / GraphMemoryProvider（未来，Layer 6）。
    """

    def store(self) -> MemoryStore:
        """持久化后端（Markdown truth source，总在）。"""
        ...

    def retriever(self) -> Retriever:
        """检索后端（HybridRetriever，组合 optional Vector/Graph adapter）。"""
        ...

    def manager(self) -> MemoryManager:
        """四操作编排（Encode/Associate/Consolidate/Reconsolidate）。"""
        ...
