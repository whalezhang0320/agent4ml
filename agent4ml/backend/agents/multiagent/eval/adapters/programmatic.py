"""L3 ProgrammaticAdapter — programmatic eval（success_criteria_met + Wilson CI）.

设计（43 文档 §4.3.1 + §11.3 L3-3.4 + spec.md ProgrammaticAdapter Requirement）:
- 部分复用 L1 ResultSummarizer.success_criteria_met（保 floor，通过 evaluator callable 注入）
- 简单规则检查独立实现（输出格式 / artifact 完整性，MVP 阶段 evaluator 内含）
- 复用 L2 _wilson_ci（import from evolution/promotion_gate，INV-20 小样本友好）
- 复用 L2 Evaluator Protocol（evaluate(artifact, task) → bool）
- task 超时/异常 skip（不补抽，R3.5 同 L2 PromotionGate pattern）
- 全 task 失败返 success=False
- health_check: evaluator 非 None 时 True
"""
from __future__ import annotations

from typing import Callable

from agent4ml.backend.agents.multiagent.eval.types import EvalContext
from agent4ml.backend.agents.multiagent.evolution.promotion_gate import (
    EvalResult,
    Evaluator,
    _wilson_ci,
)


class ProgrammaticAdapter:
    """programmatic eval——success_criteria_met + Wilson 95% CI.

    实现 EvalAdapter Protocol（evaluate + health_check）.
    evaluator: L2 Evaluator Protocol（evaluate(artifact, task) → bool），由 bootstrap 注入.
    MVP evaluator=None（floor eval 由 L2 PromotionGate 直接调，L3 ProgrammaticAdapter 数据驱动触发后才装配）.
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

    def health_check(self) -> bool:
        return self._evaluator is not None
