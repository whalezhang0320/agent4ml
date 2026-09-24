"""Skill eval 评估层包。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvalLayer:
    """L3 eval 组件容器，bootstrap 装配后注入 SkillManager。"""
    bridge: Any                      # RegistryEvalBridge
    judgment_analyzer: Any | None    # SkillJudgmentAnalyzer
    task_judge: Any | None           # TaskQualityJudge
    runtime_tracker: Any             # RuntimeTracker

