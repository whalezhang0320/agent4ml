"""L3 LongitudinalPairsAdapter — candidate vs baseline 同批 task 对比 + Wilson CI.

设计（43 文档 §4.3.3 + §11.2 + spec.md LongitudinalPairsAdapter Requirement）:
- task_sample 由 L2 抽样（80% 失败 + 20% 成功，L2 R3 已设计）
- candidate vs baseline 各跑 → success_criteria_met 评估（返 bool）
- Wilson 95% CI（candidate CI 下界 > baseline CI 上界 → accept 倾向，由 L2 PromotionGate.decide 判定）
- 复用 L2 _wilson_ci + L2 Evaluator Protocol
- 单 task 累计 eval ≤ 3 次（pool 淘汰逻辑在 L2，L3 仅评估，不实现淘汰）
- task 异常 skip / 全 task 失败返 success=False
- health_check: evaluator 非 None 时 True
- 与 ProgrammaticAdapter 实现独立（43 文档 §11.2 L3-3.1 决策 b：3 adapter 各自实现）
"""
from __future__ import annotations

from agent4ml.backend.agents.multiagent.eval.types import EvalContext
from agent4ml.backend.agents.multiagent.evolution.promotion_gate import (
    EvalResult,
    Evaluator,
    _wilson_ci,
)


class LongitudinalPairsAdapter:
    """longitudinal pairs eval——candidate vs baseline 同批 task 对比 + Wilson 95% CI.

    实现 EvalAdapter Protocol（evaluate + health_check）.
    与 ProgrammaticAdapter 实现独立（3 adapter 各自实现，L3-3.1 决策 b）.
    区别：method_used='longitudinal_pairs'，task_sample 由 L2 抽样（80% 失败 + 20% 成功）.
    evaluator: L2 Evaluator Protocol（evaluate(artifact, task) → bool），由 bootstrap 注入.
    MVP evaluator=None（数据驱动触发后才装配）.
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
                method_used="longitudinal_pairs",
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
                method_used="longitudinal_pairs",
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
            method_used="longitudinal_pairs",
            success=True,
        )

    def health_check(self) -> bool:
        return self._evaluator is not None
