"""DecayPolicy Protocol — 衰减策略契约。

承接 `Hezao-MemDesign-Docs/agent4ml/00-long-term-memory-foundation.md` §7.4 + §5.5
+ `48-memory-l1-base-layer.md` §4 Step 5.1。

默认实现：EbbinghausDecayPolicy（strategies/default/decay.py，Layer 2）。
可替换：LinearDecay / StepDecay / CustomDecay。

INVARIANT: lazy decay — strength 在 retrieve 时按需计算，不跑后台衰减任务（00 §5.5）。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agent4ml.backend.agents.memory.schema import MemoryTrace


@runtime_checkable
class DecayPolicy(Protocol):
    """衰减策略协议（00 §7.4）。

    默认实现 EbbinghausDecayPolicy（strategies/default/decay.py，Layer 2）。
    可替换 LinearDecay / StepDecay / CustomDecay。
    """

    def compute_strength(self, trace: MemoryTrace, now: float) -> float:
        """计算当前强度（lazy decay，retrieve 时调用）。

        00 §5.5 公式：
            strength = base_strength × (1 - decay_rate)^time_hours
                     + log(1 + access_count) × 0.1
                     + importance × 0.05
        """
        ...
