"""Skill 自进化层 Protocol 抽象 — 2b/L3 零侵入预留。

5 Protocol 定义自进化闭环的契约：
- Trigger：何时触发进化（should_trigger 产 EvolutionContext 列表）
- FailureFocuser：失败聚焦（focus 增强 EvolutionContext）
- Mutator：变异（mutate 产 candidate + diff）
- EvalBridge：eval 转接（evaluate 产 EvalResult；L2 自带 ProgrammaticEvalBridge，L3 替换）
- PromotionGate：门控（decide 产 GateDecision）

2b/L3 加 impl = 实现 Protocol + 注入 EvolutionManager，不动 2a 核心（零侵入）。
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from agent4ml.backend.agents.skill.evolution.types import (
    EvalContext,
    EvalResult,
    EvolutionContext,
    GateDecision,
)
from agent4ml.backend.agents.skill.types import SkillRecord


@runtime_checkable
class Trigger(Protocol):
    """触发器。读 store metrics/状态，产出 EvolutionContext 列表。"""

    def should_trigger(self, store: Any) -> list[EvolutionContext]: ...


@runtime_checkable
class FailureFocuser(Protocol):
    """失败聚焦。输入 raw context，输出聚焦后 + 修复方向。"""

    def focus(self, ctx: EvolutionContext, store: Any) -> EvolutionContext: ...


@runtime_checkable
class Mutator(Protocol):
    """变异器。受 budget/粒度/维度约束，产出 candidate（is_active=False）+ diff。"""

    def mutate(self, ctx: EvolutionContext, llm: Any | None) -> tuple[SkillRecord, str]: ...


@runtime_checkable
class EvalBridge(Protocol):
    """第二层 ↔ 第三层 eval 转接。L2 自带 ProgrammaticEvalBridge floor，L3 替换为 registry。"""

    def evaluate(self, ctx: EvalContext) -> EvalResult: ...


@runtime_checkable
class PromotionGate(Protocol):
    """提升门。用 EvalResult 决策 accept/reject/pending_human。"""

    def decide(
        self,
        candidate: SkillRecord,
        baseline: SkillRecord,
        eval_result: EvalResult,
    ) -> GateDecision: ...
