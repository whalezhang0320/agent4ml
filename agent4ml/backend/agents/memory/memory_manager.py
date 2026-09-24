"""MemoryManager Protocol — 四操作编排契约（工具里无 LLM）。

承接 `Hezao-MemDesign-Docs/agent4ml/00-long-term-memory-foundation.md` §5.3 + §7.4
+ `48-memory-l1-base-layer.md` §4 Step 5.5。

核心原则（00 D3）：工具里无 LLM。Encode/Associate/Consolidate/Reconsolidate
都是纯存储操作，LLM 生成的内容（如 consolidate 时的合并文本）由外部传入。

检索不在本 Protocol — retrieve 统一走 Retriever（MemoryProvider.retriever() 直给）。
retrieve 强化（lazy decay + access_count+1）放 Retriever 实现里。

默认实现：DefaultMemoryManager（strategies/default/manager.py，Layer 2）。

INVARIANT:
- 四操作无 LLM（consolidate/reconsolidate 的 merged_content/new_content 外部传入）
- Retrieve 移至 Retriever（不在本 Protocol）
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agent4ml.backend.agents.memory.schema import MemoryTrace, MemoryType


@runtime_checkable
class MemoryManager(Protocol):
    """四种原子操作编排（00 §5.3 + §7.4，Retrieve 移至 Retriever）。

    核心原则：工具里无 LLM。Encode/Associate/Consolidate/Reconsolidate
    都是纯存储操作，LLM 生成的内容（如 consolidate 时的合并文本）由外部传入。

    检索不在本 Protocol — retrieve 统一走 Retriever（MemoryProvider.retriever() 直给）。
    retrieve 强化（lazy decay + access_count+1）放 Retriever 实现里。

    默认实现 DefaultMemoryManager（strategies/default/manager.py，Layer 2）。
    """

    def encode(
        self,
        content: str,
        type: MemoryType,
        *,
        importance: float = 0.5,
        source: str | None = None,
        metadata: dict | None = None,
    ) -> MemoryTrace:
        """Encode（编码）：创建新记忆。

        00 §5.3 场景：用户说"我下周去东京出差" → encode 为 episodic 记忆。
        """
        ...

    def associate(
        self,
        trace_id_a: str,
        trace_id_b: str,
        *,
        strength: float = 0.5,
        type: str = "related",
    ) -> None:
        """Associate（关联）：建立两条记忆的关联。

        00 §5.3 场景：新记忆"东京出差"与"用户喜欢日料"建立关联。
        下次检索到其中一条时，另一条也被激活（扩散激活）。
        """
        ...

    def consolidate(self, trace_ids: list[str], merged_content: str) -> MemoryTrace:
        """Consolidate（巩固）：多条零散记忆合并为一条稳定知识。

        00 §5.3 场景：多条东京相关 episodic → 一条 semantic "用户经常去日本出差"。
        merged_content 由外部 LLM 生成后传入（工具里无 LLM）。
        """
        ...

    def reconsolidate(self, trace_id: str, new_content: str) -> MemoryTrace:
        """Reconsolidate（重编码）：更新已有记忆内容。

        00 §5.3 场景：用户说"其实改去大阪了" → 更新出差记忆内容。
        依赖 retrieve 结果作为输入（agent 需先回忆起旧记忆才能修正）。
        new_content 由外部 LLM 生成后传入（工具里无 LLM）。
        """
        ...
