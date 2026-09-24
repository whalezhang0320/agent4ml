"""L3 LLMJudgeAdapter — LLM 4 维加权评分 eval.

设计（43 文档 §4.3.2 + §11.2 L3-3.1 + spec.md LLMJudgeAdapter Requirement）:
- WEIGHTS 复用 skill TaskQualityJudge 权重值（0.50/0.35/0.05/0.10，不实现 skill Protocol——skill 是 async，L3 是 sync）
- judge_fn(artifact, task) → float [0,1]：bootstrap 注入（内部跑 specialist + LLM 4 维评分 + 加权汇总）
- 复用 L2 _wilson_ci（import from evolution/promotion_gate）
- task 异常 skip（fail-closed，不降级——降级由 Bridge 层选 programmatic adapter）
- 全 task 失败返 success=False
- llm_judge_model 默认 None（继承 lead，R2.5 同 L2 EvolutionMutator）
- health_check: judge_fn 非 None 时 True
"""
from __future__ import annotations

from typing import Any, Callable

from agent4ml.backend.agents.multiagent.eval.types import EvalContext
from agent4ml.backend.agents.multiagent.evolution.promotion_gate import (
    EvalResult,
    EvalTask,
    EvolutionArtifact,
    _wilson_ci,
)

JudgeFn = Callable[[EvolutionArtifact, EvalTask], float]


class LLMJudgeAdapter:
    """LLM-judge eval——4 维加权评分 + Wilson 95% CI.

    实现 EvalAdapter Protocol（evaluate + health_check）.
    WEIGHTS 复用 skill TaskQualityJudge 权重值（D-L3-13: 0.50*completion + 0.35*quality + 0.05*efficiency + 0.10*tool）.
    不实现 skill TaskQualityJudge Protocol（skill 是 async，L3 是 sync，与 L1 D10 一致）.
    judge_fn 由 bootstrap 注入（MVP None，数据驱动触发后才装配）.
    """

    WEIGHTS = {
        "task_completion": 0.50,
        "response_quality": 0.35,
        "efficiency": 0.05,
        "tool_usage": 0.10,
    }

    def __init__(
        self,
        judge_fn: JudgeFn | None = None,
        llm_judge_model: str | None = None,
        z_score: float = 1.96,
    ) -> None:
        self._judge_fn = judge_fn
        self._llm_judge_model = llm_judge_model
        self._z = z_score

    def evaluate(self, ctx: EvalContext) -> EvalResult:
        if self._judge_fn is None:
            return EvalResult(
                candidate_score=0.0, baseline_score=0.0,
                ci_low=0.0, ci_high=0.0, sample_size=0,
                method_used="llm_judge",
                success=False, failure_reason="no judge_fn configured",
            )

        candidate_scores: list[float] = []
        baseline_scores: list[float] = []

        for task in ctx.task_sample:
            try:
                c_score = self._judge_fn(ctx.candidate, task)
                b_score = self._judge_fn(ctx.baseline, task)
            except Exception:
                continue
            candidate_scores.append(max(0.0, min(1.0, c_score)))
            baseline_scores.append(max(0.0, min(1.0, b_score)))

        if not candidate_scores:
            return EvalResult(
                candidate_score=0.0, baseline_score=0.0,
                ci_low=0.0, ci_high=0.0, sample_size=0,
                method_used="llm_judge",
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
            method_used="llm_judge",
            success=True,
        )

    def health_check(self) -> bool:
        return self._judge_fn is not None
