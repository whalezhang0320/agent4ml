"""CaptureTrigger — CAPTURED 主动沉淀触发（补 Hermes 式）。

2a 仅手动 `/skill capture`（manual_capture）。自动信号（重复成功模式 / agent 自评）留 2b。
"""
from __future__ import annotations

from typing import Any

from agent4ml.backend.agents.skill.evolution.types import EvolutionContext


class CaptureTrigger:
    """CAPTURED 触发。2a 手动；自动信号留 2b（should_trigger 返空）。"""

    def should_trigger(self, store: Any) -> list[EvolutionContext]:
        """2a 自动信号未实现，返空。2b 加 post-execution 模式识别 + agent 自评。"""
        return []

    def manual_capture(self, pattern: str, suggested_name: str) -> EvolutionContext:
        """手动沉淀入口（/skill capture 命令调）。

        产 CAPTURED context：无 target_skill，含模式证据 + 建议 name。
        """
        return EvolutionContext(
            trigger="CAPTURE",
            evolution_type="CAPTURED",
            target_skill=None,
            capture_pattern=pattern,
            suggested_name=suggested_name,
        )
