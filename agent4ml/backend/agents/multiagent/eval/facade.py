"""L3 MultiagentProgrammaticFacade — L3 未启用时 L2 用的 facade.

设计（43 文档 §4.5 + §11.5 L3-6.1 + spec.md MultiagentProgrammaticFacade Requirement）:
- 实现 EvalBridge Protocol（evaluate + list_available_methods + health_check）
- evaluate 内部委托 L1 ResultSummarizer.success_criteria_met（通过 evaluator callable 注入）
- 接口与 OrchestrationBridge 一致（L2 调用方式统一）
- L3 未启用时 L2 PromotionGate 用此 facade（向后兼容），L3 启用时换 OrchestrationBridge
- 复用 skill ProgrammaticEvalBridge facade pattern（L3-6.1 决策 b 渐进迁移）
- 复用 L2 _wilson_ci + L2 Evaluator Protocol
- list_available_methods 返 ('programmatic',)（只有 1 个方法）
"""
from __future__ import annotations

from agent4ml.backend.agents.multiagent.eval.types import EvalContext
from agent4ml.backend.agents.multiagent.evolution.promotion_gate import (
    EvalResult,
    Evaluator,
    _wilson_ci,
)


class MultiagentProgrammaticFacade:
    """L3 未启用时 L2 用的 facade，实现 EvalBridge Protocol.

    内部调 L1 ResultSummarizer.success_criteria_met（通过 evaluator callable 注入）.
    接口与 OrchestrationBridge 一致（L2 PromotionGate.bridge 参数类型即 EvalBridge Protocol）.
    L3 未启用时 L2 用此 facade（floor eval），L3 启用时换 OrchestrationBridge.
    复用 skill ProgrammaticEvalBridge facade pattern（L3-6.1 决策 b 渐进迁移）.
    """

    def __init__(
        self,
        evaluator: Evaluator | None = None,
        z_score: float = 1.96,
    ) -> None:
        self._evaluator = evaluator
        self._z = z_score

    def evaluate(self, ctx: EvalContext) -> EvalResult:
        if self._evaluator is None:
            return EvalResult(
                candidate_score=0.0, baseline_score=0.0,
                ci_low=0.0, ci_high=0.0, sample_size=0,
                method_used="programmatic",
                success=False, failure_reason="no evaluator configured",
            )

        candidate_scores: list[float] = []
        baseline_scores: list[float] = []

        for task in ctx.task_sample:
            try:
                c_met = self._evaluator.evaluate(ctx.candidate, task)
                b_met = self._evaluator.evaluate(ctx.baseline, task)
            except Exception:
                continue
            candidate_scores.append(1.0 if c_met else 0.0)
            baseline_scores.append(1.0 if b_met else 0.0)

        if not candidate_scores:
            return EvalResult(
                candidate_score=0.0, baseline_score=0.0,
                ci_low=0.0, ci_high=0.0, sample_size=0,
                method_used="programmatic",
                success=False, failure_reason="all_tasks_failed",
            )

        c_score = sum(candidate_scores) / len(candidate_scores)
        b_score = sum(baseline_scores) / len(baseline_scores)
        c_low, c_high = _wilson_ci(c_score, len(candidate_scores), self._z)

        return EvalResult(
            candidate_score=c_score,
            baseline_score=b_score,
            ci_low=c_low,
            ci_high=c_high,
            sample_size=len(candidate_scores),
            method_used="programmatic",
            success=True,
        )

    def list_available_methods(self) -> tuple[str, ...]:
        """只有 programmatic 1 个方法（facade 不支持 llm_judge/longitudinal_pairs）."""
        return ("programmatic",)

    def health_check(self) -> bool:
        return self._evaluator is not None
