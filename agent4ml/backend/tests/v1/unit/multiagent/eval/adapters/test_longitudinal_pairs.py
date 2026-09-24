"""L3 LongitudinalPairsAdapter 单测.

测试要点（结合 L2 联动）:
- evaluator=None 返 success=False
- candidate 显著优于 baseline
- CI 计算（复用 L2 _wilson_ci）
- task 异常 skip
- 全 task 失败返 success=False
- method_used="longitudinal_pairs"
- expected_outcome 字段在 task 中可用（金标准 task）
- 返 L2 EvalResult 类型
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from agent4ml.backend.agents.multiagent.eval.adapters.longitudinal_pairs import (
    LongitudinalPairsAdapter,
)
from agent4ml.backend.agents.multiagent.eval.types import EvalContext
from agent4ml.backend.agents.multiagent.evolution.promotion_gate import (
    EvalResult,
    EvalTask,
)


def _make_artifact(version: str) -> Any:
    return SimpleNamespace(
        version=version, template_id="t", artifact_hash=f"h_{version}",
    )


def _make_ctx(tasks: int = 3, with_expected: bool = False) -> EvalContext:
    return EvalContext(
        candidate=_make_artifact("v2"),
        baseline=_make_artifact("v1"),
        task_sample=tuple(
            EvalTask(
                task_id=f"t{i}",
                goal="g",
                success_criteria="sc",
                expected_outcome="gold" if with_expected else None,
            )
            for i in range(tasks)
        ),
    )


class _CandidateWinEvaluator:
    """candidate 通过 / baseline 失败."""

    def evaluate(self, artifact: Any, task: EvalTask) -> bool:
        return artifact.version == "v2"


class _BaselineWinEvaluator:
    """baseline 通过 / candidate 失败（CI 重叠 reject 倾向）."""

    def evaluate(self, artifact: Any, task: EvalTask) -> bool:
        return artifact.version == "v1"


class _CrashEvaluator:
    def evaluate(self, artifact: Any, task: EvalTask) -> bool:
        raise RuntimeError("sandbox crash")


class _ExpectedOutcomeEvaluator:
    """用 expected_outcome 做金标准对比."""

    def evaluate(self, artifact: Any, task: EvalTask) -> bool:
        if task.expected_outcome is None:
            return False
        return artifact.version == "v2" and task.expected_outcome == "gold"


class TestLongitudinalPairsAdapter:
    def test_no_evaluator_returns_failure(self):
        adapter = LongitudinalPairsAdapter(evaluator=None)
        result = adapter.evaluate(_make_ctx())
        assert result.success is False
        assert result.failure_reason == "no evaluator configured"
        assert result.method_used == "longitudinal_pairs"

    def test_no_evaluator_health_check_false(self):
        adapter = LongitudinalPairsAdapter(evaluator=None)
        assert adapter.health_check() is False

    def test_candidate_significantly_better(self):
        """candidate 显著优于 baseline（candidate CI 下界 > baseline CI 上界倾向）."""
        adapter = LongitudinalPairsAdapter(evaluator=_CandidateWinEvaluator())
        result = adapter.evaluate(_make_ctx(tasks=10))
        assert result.success is True
        assert result.candidate_score == 1.0
        assert result.baseline_score == 0.0

    def test_baseline_better(self):
        """baseline 优于 candidate（CI 重叠 reject 倾向）."""
        adapter = LongitudinalPairsAdapter(evaluator=_BaselineWinEvaluator())
        result = adapter.evaluate(_make_ctx(tasks=5))
        assert result.candidate_score == 0.0
        assert result.baseline_score == 1.0

    def test_ci_computed(self):
        adapter = LongitudinalPairsAdapter(evaluator=_CandidateWinEvaluator())
        result = adapter.evaluate(_make_ctx(tasks=10))
        assert result.sample_size == 10
        assert 0.0 <= result.ci_low <= result.candidate_score <= result.ci_high <= 1.0

    def test_task_exception_skipped(self):
        adapter = LongitudinalPairsAdapter(evaluator=_CrashEvaluator())
        result = adapter.evaluate(_make_ctx(tasks=3))
        assert result.success is False
        assert result.failure_reason == "all_tasks_failed"

    def test_all_fail_still_success_with_samples(self):
        """全 fail 但有样本时 success=True（有统计意义）."""
        class _AllFail:
            def evaluate(self, artifact: Any, task: EvalTask) -> bool:
                return False
        adapter = LongitudinalPairsAdapter(evaluator=_AllFail())
        result = adapter.evaluate(_make_ctx(tasks=3))
        assert result.success is True
        assert result.candidate_score == 0.0
        assert result.baseline_score == 0.0

    def test_expected_outcome_field_usable(self):
        """expected_outcome 字段可用（金标准 task）."""
        adapter = LongitudinalPairsAdapter(evaluator=_ExpectedOutcomeEvaluator())
        result = adapter.evaluate(_make_ctx(tasks=3, with_expected=True))
        assert result.success is True
        assert result.candidate_score == 1.0
        assert result.baseline_score == 0.0

    def test_health_check_with_evaluator(self):
        adapter = LongitudinalPairsAdapter(evaluator=_CandidateWinEvaluator())
        assert adapter.health_check() is True

    def test_returns_l2_eval_result_type(self):
        adapter = LongitudinalPairsAdapter(evaluator=_CandidateWinEvaluator())
        result = adapter.evaluate(_make_ctx())
        assert isinstance(result, EvalResult)
