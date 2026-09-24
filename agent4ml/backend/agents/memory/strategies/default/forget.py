"""Composite 遗忘策略（TTL + strength，两规则）。

承接 `Hezao-MemDesign-Docs/agent4ml/00-long-term-memory-foundation.md` §7.4
+ `49-memory-l2-default-strategies.md` §4 Step 3 + §5.3 + B3。

00 §7.4 两规则组合（B3：矛盾解决走 reconsolidate/consolidate，不在此策略）：
1. TTL 过期：long time no access → forget
2. strength 阈值：衰减到 strength_threshold 以下 → forget

should_forget 检查规则 1+2（自动遗忘）。矛盾解决走 reconsolidate/consolidate（B3）。

INVARIANT：
- 两规则遗忘（B3）：should_forget 检查 TTL + strength；无 resolve_conflict
- 依赖 DecayPolicy 计算当前 strength（规则 2）
- 参数从 get_memory_config().forget 取（runtime 可切），缺省回退 _constants.FORGET_THRESHOLDS
"""

from __future__ import annotations

from agent4ml.backend.agents.memory.config import get_memory_config
from agent4ml.backend.agents.memory.schema import MemoryTrace
from agent4ml.backend.agents.memory.strategies.default._constants import FORGET_THRESHOLDS
from agent4ml.backend.agents.memory.strategies.default.decay import EbbinghausDecayPolicy


class CompositeForgetPolicy:
    """Composite 遗忘策略（00 §7.4 + 49 B3）。

    两规则组合（should_forget 检查）：
    1. TTL 过期：now - last_accessed > ttl_hours × 3600
    2. strength 阈值：compute_strength(trace, now) < strength_threshold

    B3 决策：不包含矛盾解决（resolve_conflict 已删）。矛盾解决走 reconsolidate
    （单条内容更新）或 consolidate（多条合并 + 旧标记 forgotten），由 manager 调用。

    依赖 DecayPolicy 计算当前 strength（规则 2）。
    参数从 get_memory_config().forget 取（runtime 可切），缺省回退 _constants.FORGET_THRESHOLDS。
    """

    def __init__(self, decay_policy: EbbinghausDecayPolicy | None = None) -> None:
        """初始化。

        Args:
            decay_policy: 衰减策略（用于规则 2 strength 计算）。None 时内部构造默认 EbbinghausDecayPolicy。
        """
        self._decay_policy = decay_policy or EbbinghausDecayPolicy()

    def should_forget(self, trace: MemoryTrace, now: float) -> bool:
        """是否应遗忘该记忆（自动规则 1+2）。

        Args:
            trace: 记忆痕迹（不修改）
            now: 当前 unix timestamp

        Returns:
            True 表示应遗忘（TTL 过期 或 strength 低于阈值）
        """
        thresholds = self._get_thresholds()

        # 规则 1：TTL 过期（长期未访问；last_accessed<=0 用 created_at）
        last_access = trace.last_accessed if trace.last_accessed > 0 else trace.created_at
        ttl_seconds = thresholds["ttl_hours"] * 3600.0
        if (now - last_access) > ttl_seconds:
            return True

        # 规则 2：strength 低于阈值（lazy decay 计算）
        current_strength = self._decay_policy.compute_strength(trace, now)
        if current_strength < thresholds["strength_threshold"]:
            return True

        return False

    def _get_thresholds(self) -> dict:
        """取遗忘阈值（runtime config 优先，缺省回退 _constants）。"""
        config = get_memory_config()
        if hasattr(config, "forget") and config.forget:
            return {
                "strength_threshold": config.forget.get(
                    "strength_threshold", FORGET_THRESHOLDS["strength_threshold"]
                ),
                "ttl_hours": config.forget.get(
                    "ttl_hours", FORGET_THRESHOLDS["ttl_hours"]
                ),
                "conflict_window_hours": config.forget.get(
                    "conflict_window_hours", FORGET_THRESHOLDS["conflict_window_hours"]
                ),
            }
        return FORGET_THRESHOLDS
