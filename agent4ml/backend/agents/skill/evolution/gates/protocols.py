"""门控 Protocol（2b/L3 预留）。

2a 实现 ScoreDeltaGate + GitRatchet（gates/score_delta_gate.py + gates/git_ratchet.py）。
以下 Protocol 为 2b/L3 预留，impl 时实现 + 注册，零侵入：

- ChampionGate（2b）：score delta + hard_failures 双门（区分关键 vs 非关键）
- HITLGate（2b）：人审装饰器，accept 转 pending_human staging
- CompositeGate（2b）：cascading 链式，早期 reject short-circuit
- ValidationGate（L3）：held-out 重放 + longitudinal pairs，需 L3 ValidationGateEvalAdapter
- MultiJudgeGate（L3）：多 LLM judge + majority vote，需 L3 LLMJudgeEvalAdapter

所有门实现 PromotionGate Protocol（evolution/protocols.py）。
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from agent4ml.backend.agents.skill.evolution.types import EvalResult, GateDecision
from agent4ml.backend.agents.skill.types import SkillRecord


@runtime_checkable
class ChampionGateProtocol(Protocol):
    """2b：score delta + hard_failures 双门（区分关键 vs 非关键 hard_failure）。"""

    def decide(
        self, candidate: SkillRecord, baseline: SkillRecord, eval_result: EvalResult,
    ) -> GateDecision: ...


@runtime_checkable
class HITLGateProtocol(Protocol):
    """2b：人审装饰器。accept 转 pending_human，staging 待 /skill approve|reject。"""

    def decide(
        self, candidate: SkillRecord, baseline: SkillRecord, eval_result: EvalResult,
    ) -> GateDecision: ...


@runtime_checkable
class CompositeGateProtocol(Protocol):
    """2b：cascading 链式组合，早期 reject short-circuit。"""

    def decide(
        self, candidate: SkillRecord, baseline: SkillRecord, eval_result: EvalResult,
    ) -> GateDecision: ...


@runtime_checkable
class ValidationGateProtocol(Protocol):
    """L3：held-out 重放 + longitudinal pairs。需 L3 ValidationGateEvalAdapter。"""

    def decide(
        self, candidate: SkillRecord, baseline: SkillRecord, eval_result: EvalResult,
    ) -> GateDecision: ...


@runtime_checkable
class MultiJudgeGateProtocol(Protocol):
    """L3：多 LLM judge + majority vote。需 L3 LLMJudgeEvalAdapter。"""

    def decide(
        self, candidate: SkillRecord, baseline: SkillRecord, eval_result: EvalResult,
    ) -> GateDecision: ...
