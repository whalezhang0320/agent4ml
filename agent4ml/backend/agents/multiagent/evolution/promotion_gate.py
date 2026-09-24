"""PromotionGate — longitudinal pairs eval + Wilson 95% CI + hash 防环（R3）。

设计（42 文档 §7.8 + spec.md PromotionGate Requirement + R3）:
- evaluate：candidate vs baseline 各跑 task_sample → 用 L1 ResultSummarizer.success_criteria_met 评估 → Wilson 95% CI
- decide：hash 命中近 5 版 → REJECT / candidate CI 下界 > baseline CI 上界 → ACCEPT / 否则 REJECT
- _wilson_ci：Wilson score interval（z=1.96，小样本友好，p=0/1 不退化）
- eval 整体超时 30min（R3.5）
- MVP eval 来源：L1 ResultSummarizer.success_criteria_met（floor eval）
- task 累计 ≤ 3 次（防过拟合，R3.4）
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from agent4ml.backend.agents.multiagent.evolution.types import (
    EvolutionArtifact,
    PromotionDecision,
)

if TYPE_CHECKING:
    from agent4ml.backend.agents.multiagent.eval.bridge import EvalBridge
    from agent4ml.backend.agents.multiagent.eval.types import EvalContext


@dataclass(frozen=True)
class EvalTask:
    """单个评估 task——一个历史 specialist 调用记录（L2 演化时抽样传入）."""

    task_id: str
    goal: str
    success_criteria: str
    sandbox_id: str | None = None
    context_snapshot_ref: str | None = None
    expected_outcome: str | None = None


@dataclass(frozen=True)
class EvalResult:
    """L3 evaluate 返回的评估结果（L2 自建同构，不 import skill）.

    与 skill EvalResult 同构但独立（skill 含 SkillRecord，L2 含 EvolutionArtifact）.
    """

    candidate_score: float
    baseline_score: float
    ci_low: float
    ci_high: float
    sample_size: int
    method_used: str
    raw_data_ref: str | None = None
    success: bool = True
    failure_reason: str | None = None


class Evaluator(Protocol):
    """评估器抽象（调 L1 ResultSummarizer.success_criteria_met 跑 candidate vs baseline）.

    evaluate(artifact, task) → bool：跑 artifact 在 task 上，返 success_criteria_met.
    """

    def evaluate(self, artifact: EvolutionArtifact, task: EvalTask) -> bool: ...


class PromotionGate:
    """longitudinal pairs eval + Wilson 95% CI + hash 防环（R3）.

    INVARIANT:
    - hash 命中近 5 版 → REJECT（防震，INV-7）
    - candidate CI 下界 > baseline CI 上界 → ACCEPT（INV-24）
    - Wilson score interval（z=1.96，小样本友好，p=0/1 不退化，INV-20）
    - eval 整体超时 30min → 中断 + 保持旧 is_active（INV-22）
    - task 累计 ≤ 3 次（防过拟合，R3.4）
    - MVP eval 来源：L1 ResultSummarizer.success_criteria_met（floor eval）
    """

    def __init__(
        self,
        evaluator: Evaluator | None = None,
        version_dag: Any | None = None,
        eval_timeout_seconds: float = 1800.0,  # 30min
        eval_sample_min: int = 10,
        eval_sample_max: int = 15,
        eval_task_max_reuse: int = 3,
        z_score: float = 1.96,
        bridge: "EvalBridge | None" = None,
    ) -> None:
        self._evaluator = evaluator
        self._version_dag = version_dag
        self._eval_timeout = eval_timeout_seconds
        self._sample_min = eval_sample_min
        self._sample_max = eval_sample_max
        self._task_max_reuse = eval_task_max_reuse
        self._z = z_score
        self._bridge = bridge
        # task 累计使用次数（防过拟合，R3.4）
        self._task_use_count: dict[str, int] = {}

    def evaluate(
        self,
        candidate: EvolutionArtifact,
        baseline: EvolutionArtifact,
        task_sample: list[EvalTask],
    ) -> EvalResult:
        """longitudinal pairs eval：candidate vs baseline 各跑 task_sample.

        过滤累计超 _task_max_reuse 的 task（防过拟合）.
        超时 → 返 EvalResult(success=False, failure_reason='overall_timeout').
        """
        # L3 bridge 分发：bridge 非 None 时调 bridge.evaluate(ctx)，否则既有 floor eval
        if self._bridge is not None:
            from agent4ml.backend.agents.multiagent.eval.types import EvalContext
            ctx = EvalContext(
                candidate=candidate,
                baseline=baseline,
                task_sample=tuple(task_sample),
            )
            return self._bridge.evaluate(ctx)

        if self._evaluator is None:
            return EvalResult(
                candidate_score=0.0, baseline_score=0.0,
                ci_low=0.0, ci_high=0.0, sample_size=0,
                method_used="programmatic_floor",
                success=False, failure_reason="no evaluator configured",
            )

        # 过滤累计超 max_reuse 的 task
        filtered = [t for t in task_sample if self._task_use_count.get(t.task_id, 0) < self._task_max_reuse]
        if not filtered:
            return EvalResult(
                candidate_score=0.0, baseline_score=0.0,
                ci_low=0.0, ci_high=0.0, sample_size=0,
                method_used="programmatic_floor",
                success=False, failure_reason="insufficient_tasks",
            )

        start = time.time()
        candidate_scores: list[float] = []
        baseline_scores: list[float] = []

        for task in filtered:
            # 超时检查（执行前）
            if time.time() - start > self._eval_timeout:
                return EvalResult(
                    candidate_score=0.0, baseline_score=0.0,
                    ci_low=0.0, ci_high=0.0, sample_size=len(candidate_scores),
                    method_used="programmatic_floor",
                    success=False, failure_reason="overall_timeout",
                )
            # 累计使用次数 +1
            self._task_use_count[task.task_id] = self._task_use_count.get(task.task_id, 0) + 1
            try:
                c_met = self._evaluator.evaluate(candidate, task)
                b_met = self._evaluator.evaluate(baseline, task)
            except Exception:
                # 单 task 跑挂 skip + 补抽（R3.5）
                continue
            candidate_scores.append(1.0 if c_met else 0.0)
            baseline_scores.append(1.0 if b_met else 0.0)

        if not candidate_scores:
            return EvalResult(
                candidate_score=0.0, baseline_score=0.0,
                ci_low=0.0, ci_high=0.0, sample_size=0,
                method_used="programmatic_floor",
                success=False, failure_reason="all_tasks_failed",
            )

        candidate_score = sum(candidate_scores) / len(candidate_scores)
        baseline_score = sum(baseline_scores) / len(baseline_scores)
        c_low, c_high = _wilson_ci(candidate_score, len(candidate_scores), self._z)
        b_low, b_high = _wilson_ci(baseline_score, len(baseline_scores), self._z)

        return EvalResult(
            candidate_score=candidate_score,
            baseline_score=baseline_score,
            ci_low=c_low,
            ci_high=c_high,
            sample_size=len(candidate_scores),
            method_used="programmatic_floor",
            success=True,
        )

    def decide(
        self,
        candidate: EvolutionArtifact,
        baseline: EvolutionArtifact,
        eval_result: EvalResult,
    ) -> PromotionDecision:
        """决策：hash 防环命中 → REJECT / candidate CI 下界 > baseline CI 上界 → ACCEPT / 否则 REJECT（INV-24）."""
        # hash 防环：candidate hash 命中近 5 版 → REJECT（INV-7）
        if self._version_dag is not None:
            if self._version_dag.hash_exists_in_recent(candidate.artifact_hash, window=5):
                return PromotionDecision.REJECT

        # eval 失败 → REJECT（保持旧 is_active，INV-13）
        if not eval_result.success:
            return PromotionDecision.FAILED

        # baseline CI 上界（用 candidate 的 baseline_score 算，因 EvalResult 只存 candidate CI）
        # 重新算 baseline CI（样本数同 candidate）
        b_low, b_high = _wilson_ci(
            eval_result.baseline_score, eval_result.sample_size, self._z
        )

        # candidate CI 下界 > baseline CI 上界 → ACCEPT（INV-24）
        if eval_result.ci_low > b_high:
            return PromotionDecision.ACCEPT
        return PromotionDecision.REJECT


def _wilson_ci(
    score: float, sample_size: int, z: float = 1.96
) -> tuple[float, float]:
    """Wilson score interval 计算（小样本友好，p=0/1 不退化，INV-20）.

    公式：p̂ ± z·√(p̂(1-p̂)/n + z²/(4n²)) / (1+z²/n)
    z=1.96 对应 95% CI.
    """
    if sample_size == 0:
        return 0.0, 0.0
    n = sample_size
    p_hat = max(0.0, min(1.0, score))
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / denom
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return low, high
