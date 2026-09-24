"""L3 SpecialistEvalRegistry + EvalAdapter Protocol 单测.

测试要点（结合 L2 联动）:
- EvalAdapter runtime_checkable（mock 实现通过 isinstance）
- SpecialistEvalRegistry register/get/list_methods
- 未知 method 返 None
- 重复注册覆盖
- mock adapter 契约验证（evaluate 返 L2 EvalResult / health_check 返 bool）
- 空 registry list_methods 返空 tuple
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from agent4ml.backend.agents.multiagent.eval.registry import (
    EvalAdapter,
    SpecialistEvalRegistry,
)
from agent4ml.backend.agents.multiagent.eval.types import EvalContext
from agent4ml.backend.agents.multiagent.evolution.promotion_gate import (
    EvalResult,
    EvalTask,
)


def _make_ctx() -> EvalContext:
    return EvalContext(
        candidate=SimpleNamespace(version="v2", template_id="t", artifact_hash="h2"),
        baseline=SimpleNamespace(version="v1", template_id="t", artifact_hash="h1"),
        task_sample=(EvalTask(task_id="t1", goal="g", success_criteria="sc"),),
    )


class _MockAdapter:
    """Mock EvalAdapter 实现."""

    def __init__(self, score: float = 0.8, healthy: bool = True) -> None:
        self._score = score
        self._healthy = healthy

    def evaluate(self, ctx: EvalContext) -> EvalResult:
        return EvalResult(
            candidate_score=self._score,
            baseline_score=0.5,
            ci_low=0.6,
            ci_high=0.9,
            sample_size=10,
            method_used="mock",
            success=True,
        )

    def health_check(self) -> bool:
        return self._healthy


class _IncompleteAdapter:
    """缺少 health_check 的不完整实现（不应通过 isinstance）."""

    def evaluate(self, ctx: EvalContext) -> EvalResult: ...


# ── EvalAdapter Protocol 测试 ────────────────────────────


class TestEvalAdapterProtocol:
    def test_runtime_checkable_isinstance(self):
        assert isinstance(_MockAdapter(), EvalAdapter)

    def test_runtime_checkable_rejects_incomplete(self):
        assert not isinstance(_IncompleteAdapter(), EvalAdapter)

    def test_runtime_checkable_rejects_plain_object(self):
        assert not isinstance(object(), EvalAdapter)


# ── SpecialistEvalRegistry 测试 ──────────────────────────


class TestSpecialistEvalRegistry:
    def test_empty_list_methods(self):
        registry = SpecialistEvalRegistry()
        assert registry.list_methods() == ()

    def test_get_unknown_returns_none(self):
        registry = SpecialistEvalRegistry()
        assert registry.get("programmatic") is None

    def test_register_and_get(self):
        registry = SpecialistEvalRegistry()
        adapter = _MockAdapter()
        registry.register("programmatic", adapter)
        assert registry.get("programmatic") is adapter
        assert registry.list_methods() == ("programmatic",)

    def test_register_multiple_preserves_order(self):
        registry = SpecialistEvalRegistry()
        registry.register("programmatic", _MockAdapter())
        registry.register("llm_judge", _MockAdapter())
        registry.register("longitudinal_pairs", _MockAdapter())
        assert registry.list_methods() == (
            "programmatic",
            "llm_judge",
            "longitudinal_pairs",
        )

    def test_register_duplicate_overwrites(self):
        """重复注册覆盖（后注册的生效）."""
        registry = SpecialistEvalRegistry()
        first = _MockAdapter(score=0.7)
        second = _MockAdapter(score=0.9)
        registry.register("programmatic", first)
        registry.register("programmatic", second)
        assert registry.get("programmatic") is second
        assert registry.list_methods() == ("programmatic",)

    def test_mock_adapter_evaluate_returns_l2_eval_result(self):
        """mock adapter evaluate 返 L2 EvalResult（跨模块类型联动）."""
        registry = SpecialistEvalRegistry()
        registry.register("programmatic", _MockAdapter(score=0.85))
        adapter = registry.get("programmatic")
        assert adapter is not None
        result = adapter.evaluate(_make_ctx())
        assert isinstance(result, EvalResult)
        assert result.candidate_score == 0.85
        assert result.success is True

    def test_mock_adapter_health_check_returns_bool(self):
        registry = SpecialistEvalRegistry()
        adapter = _MockAdapter(healthy=True)
        registry.register("llm_judge", adapter)
        assert adapter.health_check() is True
