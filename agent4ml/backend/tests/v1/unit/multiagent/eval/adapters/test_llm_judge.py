"""L3 LLMJudgeAdapter 单测.

测试要点（结合 L2 联动）:
- judge_fn=None 返 success=False
- candidate 高分 / baseline 低分
- CI 计算（复用 L2 _wilson_ci）
- task 异常 skip
- score 被 clamp 到 [0, 1]
- WEIGHTS 值正确（0.50/0.35/0.05/0.10，复用 skill TaskQualityJudge）
- method_used="llm_judge"
- 返 L2 EvalResult 类型
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from agent4ml.backend.agents.multiagent.eval.adapters.llm_judge import (
    LLMJudgeAdapter,
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


def _make_ctx(tasks: int = 3) -> EvalContext:
    return EvalContext(
        candidate=_make_artifact("v2"),
        baseline=_make_artifact("v1"),
        task_sample=tuple(
            EvalTask(task_id=f"t{i}", goal="g", success_criteria="sc")
            for i in range(tasks)
        ),
    )


class _HighScoreJudge:
    """candidate 返 0.9 / baseline 返 0.3."""

    def __call__(self, artifact: Any, task: EvalTask) -> float:
        return 0.9 if artifact.version == "v2" else 0.3


class _CrashJudge:
    """总是抛异常."""

    def __call__(self, artifact: Any, task: EvalTask) -> float:
        raise RuntimeError("LLM call failed")


class _OverflowJudge:
    """返超范围分数（测 clamp）."""

    def __call__(self, artifact: Any, task: EvalTask) -> float:
        return 1.5 if artifact.version == "v2" else -0.2


class TestLLMJudgeAdapter:
    def test_weights_match_skill_task_quality_judge(self):
        """WEIGHTS 复用 skill TaskQualityJudge 权重值（D-L3-13）."""
        assert LLMJudgeAdapter.WEIGHTS == {
            "task_completion": 0.50,
            "response_quality": 0.35,
            "efficiency": 0.05,
            "tool_usage": 0.10,
        }

    def test_no_judge_fn_returns_failure(self):
        adapter = LLMJudgeAdapter(judge_fn=None)
        result = adapter.evaluate(_make_ctx())
        assert result.success is False
        assert result.failure_reason == "no judge_fn configured"
        assert result.method_used == "llm_judge"

    def test_no_judge_fn_health_check_false(self):
        adapter = LLMJudgeAdapter(judge_fn=None)
        assert adapter.health_check() is False

    def test_candidate_higher_than_baseline(self):
        adapter = LLMJudgeAdapter(judge_fn=_HighScoreJudge())
        result = adapter.evaluate(_make_ctx(tasks=5))
        assert result.success is True
        assert result.candidate_score == 0.9
        assert result.baseline_score == 0.3
        assert result.method_used == "llm_judge"

    def test_ci_computed(self):
        adapter = LLMJudgeAdapter(judge_fn=_HighScoreJudge())
        result = adapter.evaluate(_make_ctx(tasks=10))
        assert result.sample_size == 10
        assert 0.0 <= result.ci_low <= result.candidate_score <= result.ci_high <= 1.0

    def test_task_exception_skipped(self):
        adapter = LLMJudgeAdapter(judge_fn=_CrashJudge())
        result = adapter.evaluate(_make_ctx(tasks=3))
        assert result.success is False
        assert result.failure_reason == "all_tasks_failed"

    def test_score_clamped_to_unit_range(self):
        """score 被 clamp 到 [0, 1]（judge_fn 返超范围值）."""
        adapter = LLMJudgeAdapter(judge_fn=_OverflowJudge())
        result = adapter.evaluate(_make_ctx(tasks=3))
        assert result.success is True
        assert result.candidate_score == 1.0  # 1.5 clamped to 1.0
        assert result.baseline_score == 0.0  # -0.2 clamped to 0.0

    def test_health_check_with_judge_fn(self):
        adapter = LLMJudgeAdapter(judge_fn=_HighScoreJudge())
        assert adapter.health_check() is True

    def test_returns_l2_eval_result_type(self):
        adapter = LLMJudgeAdapter(judge_fn=_HighScoreJudge())
        result = adapter.evaluate(_make_ctx())
        assert isinstance(result, EvalResult)

    def test_llm_judge_model_default_none(self):
        """llm_judge_model 默认 None（继承 lead，R2.5 同 L2 EvolutionMutator）."""
        adapter = LLMJudgeAdapter(judge_fn=_HighScoreJudge())
        assert adapter._llm_judge_model is None
