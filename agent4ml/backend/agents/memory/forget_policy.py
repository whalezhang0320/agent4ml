"""ForgetPolicy Protocol — 遗忘策略契约。

承接 `Hezao-MemDesign-Docs/agent4ml/00-long-term-memory-foundation.md` §7.4
+ `48-memory-l1-base-layer.md` §4 Step 5.2 + `49-memory-l2-default-strategies.md` B3。

默认实现：CompositeForgetPolicy（strategies/default/forget.py，Layer 2）。
组合：TTL 过期 + strength 阈值（两规则）。
可替换：TTLOnly / StrengthOnly。

B3 决策：矛盾解决不在本策略（走 reconsolidate 单条内容更新或 consolidate 多条合并 +
旧标记 forgotten，由 manager 在 Phase 2 LLM 决策时调用）。ForgetPolicy 职责单一化：
只判 should_forget，不解决矛盾。

INVARIANT: Protocol 纯契约零实现，可 mock 可替换。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agent4ml.backend.agents.memory.schema import MemoryTrace


@runtime_checkable
class ForgetPolicy(Protocol):
    """遗忘策略协议（00 §7.4 + 49 B3）。

    默认实现 CompositeForgetPolicy（strategies/default/forget.py，Layer 2）。
    组合：TTL 过期 + strength 阈值（两规则，should_forget 检查）。

    B3 决策：不包含矛盾解决（resolve_conflict 已删）。矛盾解决走 reconsolidate
    （单条内容更新）或 consolidate（多条合并 + 旧标记 forgotten），由 manager 调用。
    """

    def should_forget(self, trace: MemoryTrace, now: float) -> bool:
        """是否应遗忘该记忆（TTL 过期 或 strength 低于阈值）。"""
        ...
