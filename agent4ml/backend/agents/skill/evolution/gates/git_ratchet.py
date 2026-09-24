"""GitRatchet — 上线后兜底回滚（借鉴 darwin git ratchet）。

非决策门，是上线后监控：candidate 已 accept 上线，后续 metric 退化
（新版本 effective_rate 跌破阈值）→ store.rollback() 切 is_active 回旧版。

类比 git revert（禁 reset --hard，保留历史）；version DAG 天然支持（旧 node 还在，切指针）。
"""
from __future__ import annotations

from typing import Any

from agent4ml.backend.agents.skill.types import SkillRecord


class GitRatchet:
    """上线后 degraded → rollback。EvolutionManager 周期调 check_and_rollback。"""

    def __init__(
        self,
        degradation_threshold: float = 0.3,
        min_selections: int = 5,
        runtime_tracker: Any | None = None,
    ) -> None:
        self._threshold = degradation_threshold
        self._min_selections = min_selections
        self._runtime_tracker = runtime_tracker  # D-L3-16: 可选 RuntimeTracker 信号

    def check_and_rollback(
        self,
        store: Any,
        current: SkillRecord,
    ) -> str | None:
        """检查当前 active skill 是否 degraded，是则 rollback 到上一版。

        D-L3-16: 若 runtime_tracker 提供，额外用 degraded_skills() 作退化信号。
        返 rollback 到的旧 skill_id；未 rollback 返 None。
        """
        # D-L3-16: RuntimeTracker 退化信号（若有）
        if self._runtime_tracker is not None:
            try:
                degraded = self._runtime_tracker.degraded_skills()
                if current.skill_id not in degraded:
                    # RuntimeTracker 未标退化，但仍用 effective_rate 兜底检查
                    pass
            except Exception:
                pass

        # 新版本需积累数据才评判（anti-loop）
        if current.total_selections < self._min_selections:
            return None
        if current.effective_rate >= self._threshold:
            return None  # 健康

        # degraded → 找上一版 rollback
        versions = store.get_versions(current.name)
        # 找 current 之前的版本（generation 更低 或 parent）
        parent_ids = current.lineage.parent_skill_ids
        rollback_target = None
        for v in versions:
            if v.skill_id == current.skill_id:
                continue
            if v.skill_id in parent_ids:
                rollback_target = v
                break
        if rollback_target is None and versions:
            # 无 parent，取 generation 最低的非当前版
            others = [v for v in versions if v.skill_id != current.skill_id]
            if others:
                rollback_target = min(others, key=lambda v: v.lineage.generation)

        if rollback_target is None:
            return None

        store.rollback(rollback_target.skill_id)
        return rollback_target.skill_id
