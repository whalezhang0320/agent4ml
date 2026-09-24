"""Ebbinghaus 衰减策略（lazy decay）。

承接 `Hezao-MemDesign-Docs/agent4ml/00-long-term-memory-foundation.md` §5.5
+ `49-memory-l2-default-strategies.md` §4 Step 2 + §5.1。

00 §5.5 公式：
    strength = base_strength × (1 - decay_rate)^time_hours
             + log(1 + access_count) × 0.1
             + importance × 0.05

lazy decay：compute_strength 不修改 trace，retrieve 时由 Retriever（Layer 3）调用
+ with_strength 更新。strength 钳制 [0.0, 1.0]。

INVARIANT：
- 纯计算，不持有状态，线程安全
- 参数从 get_memory_config().decay 取（runtime 可切），缺省回退 _constants.DECAY_PARAMS
- 不修改 trace（frozen 语义，返回 float）
"""

from __future__ import annotations

import math

from agent4ml.backend.agents.memory.config import get_memory_config
from agent4ml.backend.agents.memory.schema import MemoryTrace, MemoryType
from agent4ml.backend.agents.memory.strategies.default._constants import (
    DECAY_COEFFICIENTS,
    DECAY_PARAMS,
)


class EbbinghausDecayPolicy:
    """Ebbinghaus 衰减策略（00 §5.5）。

    纯计算，不持有状态，线程安全。
    参数从 get_memory_config().decay 取（runtime 可切），缺省回退 _constants.DECAY_PARAMS。
    """

    def compute_strength(self, trace: MemoryTrace, now: float) -> float:
        """计算当前强度（lazy decay，retrieve 时调用）。

        Args:
            trace: 记忆痕迹（不修改）
            now: 当前 unix timestamp

        Returns:
            当前强度 0.0~1.0（钳制）
        """
        # 1. 取衰减参数（runtime config 优先，缺省回退 _constants）
        params = self._get_decay_params(trace.type)
        base_strength = params["base_strength"]
        decay_rate = params["decay_rate"]

        # 2. time_hours（自上次访问；未访问用 created_at）
        if trace.last_accessed <= 0:
            time_hours = max(0.0, (now - trace.created_at) / 3600.0)
        else:
            time_hours = max(0.0, (now - trace.last_accessed) / 3600.0)

        # 3. Ebbinghaus 衰减项 base_strength × (1 - decay_rate)^time_hours
        decay_factor = (1.0 - decay_rate) ** time_hours
        decayed_strength = base_strength * decay_factor

        # 4. 访问强化项 log(1 + access_count) × 0.1
        access_boost = math.log(1 + trace.access_count) * DECAY_COEFFICIENTS["access_boost"]

        # 5. 重要性加成 importance × 0.05
        importance_boost = trace.importance * DECAY_COEFFICIENTS["importance_boost"]

        # 6. 合成 + 钳制 [0, 1]
        strength = decayed_strength + access_boost + importance_boost
        return max(0.0, min(1.0, strength))

    def _get_decay_params(self, type: MemoryType) -> dict:
        """取衰减参数（runtime config 优先，缺省回退 _constants）。

        runtime 可切：set_memory_config() 替换 config.decay 后立即生效。
        """
        config = get_memory_config()
        type_key = type.value if isinstance(type, MemoryType) else str(type)
        # config.decay 覆盖（若该 type 在 config 中有定义）
        if hasattr(config, "decay") and type_key in config.decay:
            return config.decay[type_key]
        # 缺省回退 _constants
        return DECAY_PARAMS[type_key]
