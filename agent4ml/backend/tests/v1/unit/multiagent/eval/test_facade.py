"""L3 MultiagentProgrammaticFacade 单测.

测试要点（结合 L2 联动）:
- 实现 EvalBridge Protocol（isinstance 检查）
- evaluator=None 返 success=False
- evaluate 委托 evaluator（跟 ProgrammaticAdapter 逻辑一致）
- list_available_methods 返 ('programmatic',)
- health_check: evaluator 非 None 时 True
- 接口与 OrchestrationBridge 一致（L2 调用方式统一）
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from agent4ml.backend.agents.multiagent.eval.bridge import EvalBridge
from agent4ml.backend.agents.multiagent.eval.facade import MultiagentProgrammaticFacade
from agent4ml.backend.agents.multiagent.eval.types import EvalContext
from agent4ml.backend.agents.multiagent.evolution.promotion_gate import (
    EvalResult,
    EvalTask,
)


def _make_ctx(tasks: int = 3) -> EvalContext:
    return EvalContext(
        candidate=SimpleNamespace(version="v2", template_id="t", artifact_hash="h2"),
        baseline=SimpleNamespace(version="v1", template_id="t", artifact_hash="h1"),
        task_sample=tuple(
            EvalTask(task_id=f"t{i}", goal="g", success_criteria="sc")
            for i in range(tasks)
        ),
    )


class _SelectiveEvaluator:
    """candidate 通过 / baseline 失败."""

    def evaluate(self, artifact: Any, task: EvalTask) -> bool:
        return artifact.version == "v2"


class _CrashEvaluator:
    def evaluate(self, artifact: Any, task: EvalTask) -> bool:
        raise RuntimeError("crash")


class TestMultiagentProgrammaticFacade:
    def test_implements_eval_bridge_protocol(self):
        """实现 EvalBridge Protocol（isinstance 检查）."""
        facade = MultiagentProgrammaticFacade(evaluator=_SelectiveEvaluator())
        assert isinstance(facade, EvalBridge)

    def test_no_evaluator_returns_failure(self):
        facade = MultiagentProgrammaticFacade(evaluator=None)
        result = facade.evaluate(_make_ctx())
        assert result.success is False
        assert result.failure_reason == "no evaluator configured"

    def test_no_evaluator_health_check_false(self):
        facade = MultiagentProgrammaticFacade(evaluator=None)
        assert facade.health_check() is False

    def test_evaluate_delegates_to_evaluator(self):
        """evaluate 委托 evaluator（candidate 通过 / baseline 失败）."""
        facade = MultiagentProgrammaticFacade(evaluator=_SelectiveEvaluator())
        result = facade.evaluate(_make_ctx(tasks=5))
        assert result.success is True
        assert result.candidate_score == 1.0
        assert result.baseline_score == 0.0

    def test_task_exception_skipped(self):
        facade = MultiagentProgrammaticFacade(evaluator=_CrashEvaluator())
        result = facade.evaluate(_make_ctx(tasks=3))
        assert result.success is False
        assert result.failure_reason == "all_tasks_failed"

    def test_list_available_methods(self):
        """list_available_methods 返 ('programmatic',)（只有 1 个方法）."""
        facade = MultiagentProgrammaticFacade(evaluator=_SelectiveEvaluator())
        assert facade.list_available_methods() == ("programmatic",)

    def test_health_check_with_evaluator(self):
        facade = MultiagentProgrammaticFacade(evaluator=_SelectiveEvaluator())
        assert facade.health_check() is True

    def test_returns_l2_eval_result_type(self):
        """返 L2 EvalResult 类型（跨模块类型联动）."""
        facade = MultiagentProgrammaticFacade(evaluator=_SelectiveEvaluator())
        result = facade.evaluate(_make_ctx())
        assert isinstance(result, EvalResult)

    def test_interface_consistent_with_orchestration_bridge(self):
        """接口与 OrchestrationBridge 一致（L2 调用方式统一）."""
        facade = MultiagentProgrammaticFacade(evaluator=_SelectiveEvaluator())
        # 3 个方法都有（evaluate / list_available_methods / health_check）
        assert callable(facade.evaluate)
        assert callable(facade.list_available_methods)
        assert callable(facade.health_check)
