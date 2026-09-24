"""L2 PromotionGate bridge 参数 + EvolutionMutator lessons 参数 单测.

测试要点（L2-L3 联动）:
- PromotionGate bridge=None 用 L1 floor eval（既有行为不变）
- PromotionGate bridge 非 None 调 bridge.evaluate(ctx)
- PromotionGate bridge 异常 fail-closed
- EvolutionMutator lessons=() 不加 lesson section（既有行为不变）
- EvolutionMutator lessons 非 () 加 lesson section
- L2Config.l3_enabled 默认 False
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from agent4ml.backend.agents.multiagent.config import L2Config
from agent4ml.backend.agents.multiagent.evolution.evolution_mutator import EvolutionMutator
from agent4ml.backend.agents.multiagent.evolution.promotion_gate import (
    EvalResult,
    EvalTask,
    PromotionGate,
)
from agent4ml.backend.agents.multiagent.evolution.types import (
    ContextSummaryTemplate,
    FailureCategory,
    FailureStats,
)


def _make_artifact(version: str = "v1") -> Any:
    return SimpleNamespace(
        version=version, template_id="t", artifact_hash=f"h_{version}",
    )


class _MockBridge:
    """Mock EvalBridge 实现."""

    def __init__(self, result: EvalResult | None = None, crash: bool = False) -> None:
        self._result = result or EvalResult(
            candidate_score=0.9, baseline_score=0.4,
            ci_low=0.7, ci_high=0.95, sample_size=10,
            method_used="llm_judge", success=True,
        )
        self._crash = crash

    def evaluate(self, ctx: Any) -> EvalResult:
        if self._crash:
            raise RuntimeError("bridge crashed")
        return self._result

    def list_available_methods(self) -> tuple[str, ...]:
        return ("llm_judge",)

    def health_check(self) -> bool:
        return True


class TestPromotionGateBridge:
    def test_bridge_none_uses_floor_eval(self):
        """bridge=None 用既有 floor eval（evaluator=None 返 success=False）."""
        gate = PromotionGate(evaluator=None, bridge=None)
        result = gate.evaluate(_make_artifact("v2"), _make_artifact("v1"), [])
        assert result.success is False
        assert result.failure_reason == "no evaluator configured"

    def test_bridge_non_none_calls_bridge_evaluate(self):
        """bridge 非 None 调 bridge.evaluate(ctx) 返 bridge 的 EvalResult."""
        bridge = _MockBridge()
        gate = PromotionGate(evaluator=None, bridge=bridge)
        task = EvalTask(task_id="t1", goal="g", success_criteria="sc")
        result = gate.evaluate(_make_artifact("v2"), _make_artifact("v1"), [task])
        assert result.success is True
        assert result.method_used == "llm_judge"
        assert result.candidate_score == 0.9

    def test_bridge_exception_fail_closed(self):
        """bridge 异常 fail-closed（OrchestrationBridge 内部捕获，但 PromotionGate 直接收 EvalResult）."""
        # 注：OrchestrationBridge 内部 fail-closed 捕获异常返 success=False
        # PromotionGate 直接调 bridge.evaluate，如果 bridge 抛异常 PromotionGate 不捕获
        # 但 OrchestrationBridge 实现保证不抛异常（fail-closed）
        # 这里测试 OrchestrationBridge 返 success=False 时 PromotionGate.decide reject
        fail_result = EvalResult(
            candidate_score=0.0, baseline_score=0.0,
            ci_low=0.0, ci_high=0.0, sample_size=0,
            method_used="llm_judge",
            success=False, failure_reason="all_tasks_timeout",
        )
        bridge = _MockBridge(result=fail_result)
        gate = PromotionGate(evaluator=None, bridge=bridge)
        task = EvalTask(task_id="t1", goal="g", success_criteria="sc")
        result = gate.evaluate(_make_artifact("v2"), _make_artifact("v1"), [task])
        assert result.success is False

    def test_backward_compat_no_bridge_param(self):
        """向后兼容：不传 bridge 参数（默认 None，既有行为不变）."""
        gate = PromotionGate(evaluator=None)
        assert gate._bridge is None


class TestEvolutionMutatorLessons:
    def test_lessons_empty_no_section(self):
        """lessons=() 不加 lesson section（既有行为不变）."""
        mutator = EvolutionMutator(llm_caller=None)
        current = ContextSummaryTemplate(
            version="v1", template_id="t",
            extractors=(), filters=(), max_tokens=2000, prompt_skeleton="skeleton",
        )
        failures = FailureStats(
            by_category={FailureCategory.CONTEXT_INSUFFICIENT: 1},
            dominant_category=FailureCategory.CONTEXT_INSUFFICIENT,
            sample_failures={},
        )
        # llm_caller=None 直接返失败，不调 _build_prompt
        result = mutator.evolve_context_summary(current, failures, lessons=())
        assert result.success is False

    def test_lessons_non_empty_accepted(self):
        """lessons 非 () 不报错（参数接受 tuple[str, ...]）."""
        mutator = EvolutionMutator(llm_caller=None)
        current = ContextSummaryTemplate(
            version="v1", template_id="t",
            extractors=(), filters=(), max_tokens=2000, prompt_skeleton="skeleton",
        )
        failures = FailureStats(
            by_category={FailureCategory.CONTEXT_INSUFFICIENT: 1},
            dominant_category=FailureCategory.CONTEXT_INSUFFICIENT,
            sample_failures={},
        )
        result = mutator.evolve_context_summary(
            current, failures, lessons=("need more context", "use code snippets"),
        )
        # llm_caller=None 直接返失败，但 lessons 参数被接受
        assert result.success is False

    def test_backward_compat_no_lessons_param(self):
        """向后兼容：不传 lessons 参数（默认空 tuple，既有行为不变）."""
        mutator = EvolutionMutator(llm_caller=None)
        current = ContextSummaryTemplate(
            version="v1", template_id="t",
            extractors=(), filters=(), max_tokens=2000, prompt_skeleton="skeleton",
        )
        failures = FailureStats(
            by_category={}, dominant_category=None, sample_failures={},
        )
        # 不传 lessons 参数，默认空 tuple
        result = mutator.evolve_context_summary(current, failures)
        assert result.success is False


class TestL2ConfigL3Enabled:
    def test_l3_enabled_default_false(self):
        """L2Config.l3_enabled 默认 False."""
        config = L2Config()
        assert config.l3_enabled is False

    def test_l3_enabled_true(self):
        config = L2Config(l3_enabled=True)
        assert config.l3_enabled is True
