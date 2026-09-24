"""ScoreDeltaGate — 基础门（零 LLM，用 EvalResult）。

规则：candidate.score > baseline.score AND 无 hard_failure → accept；else reject。
D7：门用 EvalResult 非 LLM 自评（SkillLens 46.4% 不可靠）。
"""
from __future__ import annotations

from agent4ml.backend.agents.skill.evolution.types import EvalResult, GateDecision
from agent4ml.backend.agents.skill.types import SkillRecord


class ScoreDeltaGate:
    """candidate score > baseline score + 无 hard_failure → accept。

    min_delta：最小 score 增量（默认 0，严格 > 即可）。
    """

    def __init__(self, min_delta: float = 0.0) -> None:
        self._min_delta = min_delta

    def decide(
        self,
        candidate: SkillRecord,
        baseline: SkillRecord,
        eval_result: EvalResult,
    ) -> GateDecision:
        # hard_failure → reject（candidate 改坏了）
        if eval_result.hard_failures:
            return GateDecision(
                recommendation="reject",
                reason=f"hard_failures: {eval_result.hard_failures}",
            )
        # CAPTURED 无 baseline（新 skill）→ score>0 即 accept（无 delta 对比）
        if baseline is None:
            if eval_result.score > 0:
                return GateDecision(
                    recommendation="accept",
                    reason=f"CAPTURED score={eval_result.score:.2f}",
                    new_version_id=candidate.skill_id,
                )
            return GateDecision(recommendation="reject", reason="CAPTURED score=0")

        # FIX/DERIVED：candidate score > baseline score + min_delta
        # baseline score 从 evidence 的 baseline_pass 比例推断
        baseline_score = self._baseline_score(eval_result)
        if eval_result.score > baseline_score + self._min_delta:
            return GateDecision(
                recommendation="accept",
                reason=f"candidate={eval_result.score:.2f} > baseline={baseline_score:.2f} + delta={self._min_delta}",
                new_version_id=candidate.skill_id,
            )
        return GateDecision(
            recommendation="reject",
            reason=f"candidate={eval_result.score:.2f} <= baseline={baseline_score:.2f} + delta={self._min_delta}",
        )

    @staticmethod
    def _baseline_score(eval_result: EvalResult) -> float:
        """从 evidence 的 baseline_pass 比例推断 baseline score。无 evidence 返 0。"""
        if not eval_result.evidence:
            return 0.0
        passed = sum(1 for e in eval_result.evidence if e.baseline_pass)
        return passed / len(eval_result.evidence)
