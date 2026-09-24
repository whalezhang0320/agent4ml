"""L3 ProgrammaticAdapter 单测.

测试要点（结合 L2 联动）:
- evaluator=None 返 success=False
- candidate 全 success → high score / baseline 全 fail → low score
- CI 计算（复用 L2 _wilson_ci）
- task 异常 skip（不补抽）
- 全 task 失败返 success=False
- health_check: evaluator 非 None 时 True
- mock evaluator 联动 L2 Evaluator Protocol
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from agent4ml.backend.agents.multiagent.eval.adapters.programmatic import (
    ProgrammaticAdapter,
)
from agent4ml.backend.agents.multiagent.eval.types import EvalContext
from agent4ml.backend.agents.multiagent.evolution.promotion_gate import (
    EvalResult,
    EvalTask,
    Evaluator,
)


def _make_artifact(version: str) -> Any:
    return SimpleNamespace(
        version=version, template_id="t", artifact_hash=f"h_{version}",
    )


def _make_ctx(tasks: int = 3) -> EvalContext:
    return EvalContext(
        candidate=_make_artifact("v2"),
        baseline=_make_artifact("v1"),
        task_sample=tuple(
            EvalTask(task_id=f"t{i}", goal="g", success_criteria="sc")
            for i in range(tasks)
        ),
    )


class _PassEvaluator:
    """全部通过的 evaluator."""

    def evaluate(self, artifact: Any, task: EvalTask) -> bool:
        return True


class _FailEvaluator:
    """全部失败的 evaluator."""

    def evaluate(self, artifact: Any, task: EvalTask) -> bool:
        return False


class _SelectiveEvaluator:
    """candidate 通过 / baseline 失败."""

    def evaluate(self, artifact: Any, task: EvalTask) -> bool:
        return artifact.version == "v2"


class _CrashEvaluator:
    """总是抛异常的 evaluator."""

    def evaluate(self, artifact: Any, task: EvalTask) -> bool:
        raise RuntimeError("sandbox crash")


class TestProgrammaticAdapter:
    def test_no_evaluator_returns_failure(self):
        adapter = ProgrammaticAdapter(evaluator=None)
        result = adapter.evaluate(_make_ctx())
        assert result.success is False
        assert result.failure_reason == "no evaluator configured"

    def test_no_evaluator_health_check_false(self):
        adapter = ProgrammaticAdapter(evaluator=None)
        assert adapter.health_check() is False

    def test_candidate_all_pass_high_score(self):
        adapter = ProgrammaticAdapter(evaluator=_SelectiveEvaluator())
        result = adapter.evaluate(_make_ctx(tasks=5))
        assert result.success is True
        assert result.candidate_score == 1.0
        assert result.baseline_score == 0.0
        assert result.method_used == "programmatic"

    def test_baseline_all_pass(self):
        """baseline 全通过（candidate 全失败）."""
        class _BaselinePass:
            def evaluate(self, artifact: Any, task: EvalTask) -> bool:
                return artifact.version == "v1"
        adapter = ProgrammaticAdapter(evaluator=_BaselinePass())
        result = adapter.evaluate(_make_ctx(tasks=3))
        assert result.candidate_score == 0.0
        assert result.baseline_score == 1.0

    def test_ci_computed(self):
        """CI 计算（复用 L2 _wilson_ci，小样本不退化）."""
        adapter = ProgrammaticAdapter(evaluator=_SelectiveEvaluator())
        result = adapter.evaluate(_make_ctx(tasks=10))
        assert result.sample_size == 10
        assert 0.0 <= result.ci_low <= result.candidate_score <= result.ci_high <= 1.0

    def test_task_exception_skipped(self):
        """task 异常 skip（不补抽，R3.5）."""
        adapter = ProgrammaticAdapter(evaluator=_CrashEvaluator())
        result = adapter.evaluate(_make_ctx(tasks=3))
        assert result.success is False
        assert result.failure_reason == "all_tasks_failed"
        assert result.sample_size == 0

    def test_all_fail_returns_failure(self):
        adapter = ProgrammaticAdapter(evaluator=_FailEvaluator())
        result = adapter.evaluate(_make_ctx(tasks=3))
        assert result.success is True  # 全 fail 但有样本，success=True
        assert result.candidate_score == 0.0
        assert result.baseline_score == 0.0

    def test_health_check_with_evaluator(self):
        adapter = ProgrammaticAdapter(evaluator=_PassEvaluator())
        assert adapter.health_check() is True

    def test_returns_l2_eval_result_type(self):
        """返 L2 EvalResult 类型（跨模块类型联动）."""
        adapter = ProgrammaticAdapter(evaluator=_PassEvaluator())
        result = adapter.evaluate(_make_ctx())
        assert isinstance(result, EvalResult)
