"""L3 OrchestrationBridge 单测.

测试要点（结合 L2 + 3 adapter 联动）:
- _select_method 各分支（hint / expected_outcome / open_ended / programmatic）
- fail-closed: adapter 异常返 success=False
- adapter 未注册返 success=False
- list_available_methods 返 registry 内容
- health_check 空 registry 返 False / 非空返 True
- 实现 EvalBridge Protocol（isinstance 检查）
- mock adapter 联动 3 adapter（programmatic/llm_judge/longitudinal_pairs）
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from agent4ml.backend.agents.multiagent.eval.bridge import (
    EvalBridge,
    OrchestrationBridge,
)
from agent4ml.backend.agents.multiagent.eval.registry import SpecialistEvalRegistry
from agent4ml.backend.agents.multiagent.eval.types import EvalContext
from agent4ml.backend.agents.multiagent.evolution.promotion_gate import (
    EvalResult,
    EvalTask,
)


def _make_ctx(
    hint: str | None = None,
    with_expected: bool = False,
    task_type: str | None = None,
    tasks: int = 2,
) -> EvalContext:
    return EvalContext(
        candidate=SimpleNamespace(version="v2", template_id="t", artifact_hash="h2"),
        baseline=SimpleNamespace(version="v1", template_id="t", artifact_hash="h1"),
        task_sample=tuple(
            EvalTask(
                task_id=f"t{i}", goal="g", success_criteria="sc",
                expected_outcome="gold" if with_expected else None,
            )
            for i in range(tasks)
        ),
        eval_method_hint=hint,
        metadata={"task_type": task_type} if task_type else {},
    )


class _MockAdapter:
    """Mock adapter 返固定 EvalResult."""

    def __init__(self, method: str, success: bool = True) -> None:
        self._method = method
        self._success = success

    def evaluate(self, ctx: EvalContext) -> EvalResult:
        if not self._success:
            raise RuntimeError(f"{self._method} crashed")
        return EvalResult(
            candidate_score=0.9, baseline_score=0.4,
            ci_low=0.7, ci_high=0.95, sample_size=10,
            method_used=self._method, success=True,
        )

    def health_check(self) -> bool:
        return True


def _registry_with(*methods: str) -> SpecialistEvalRegistry:
    reg = SpecialistEvalRegistry()
    for m in methods:
        reg.register(m, _MockAdapter(m))
    return reg


class TestSelectMethod:
    def test_hint_takes_priority(self):
        bridge = OrchestrationBridge(_registry_with("longitudinal_pairs"))
        result = bridge.evaluate(_make_ctx(hint="longitudinal_pairs", with_expected=True))
        assert result.method_used == "longitudinal_pairs"

    def test_expected_outcome_selects_longitudinal(self):
        bridge = OrchestrationBridge(_registry_with("longitudinal_pairs"))
        result = bridge.evaluate(_make_ctx(with_expected=True))
        assert result.method_used == "longitudinal_pairs"

    def test_open_ended_selects_llm_judge(self):
        bridge = OrchestrationBridge(_registry_with("llm_judge"))
        result = bridge.evaluate(_make_ctx(task_type="open_ended"))
        assert result.method_used == "llm_judge"

    def test_default_selects_programmatic(self):
        bridge = OrchestrationBridge(_registry_with("programmatic"))
        result = bridge.evaluate(_make_ctx())
        assert result.method_used == "programmatic"


class TestFailClosed:
    def test_adapter_exception_returns_failure(self):
        """adapter 异常 fail-closed 返 success=False（不抛异常）."""
        reg = SpecialistEvalRegistry()
        reg.register("programmatic", _MockAdapter("programmatic", success=False))
        bridge = OrchestrationBridge(reg)
        result = bridge.evaluate(_make_ctx())
        assert result.success is False
        assert "crashed" in (result.failure_reason or "")

    def test_adapter_not_registered_returns_failure(self):
        """adapter 未注册返 success=False."""
        bridge = OrchestrationBridge(SpecialistEvalRegistry())
        result = bridge.evaluate(_make_ctx())
        assert result.success is False
        assert "not registered" in (result.failure_reason or "")


class TestProtocolAndHealth:
    def test_implements_eval_bridge_protocol(self):
        bridge = OrchestrationBridge(_registry_with("programmatic"))
        assert isinstance(bridge, EvalBridge)

    def test_list_available_methods(self):
        reg = _registry_with("programmatic", "llm_judge", "longitudinal_pairs")
        bridge = OrchestrationBridge(reg)
        assert bridge.list_available_methods() == (
            "programmatic", "llm_judge", "longitudinal_pairs",
        )

    def test_health_check_empty_registry_false(self):
        bridge = OrchestrationBridge(SpecialistEvalRegistry())
        assert bridge.health_check() is False

    def test_health_check_non_empty_registry_true(self):
        bridge = OrchestrationBridge(_registry_with("programmatic"))
        assert bridge.health_check() is True
