"""L3 EvalBridge Protocol + EvalContext 单测.

测试要点（结合 L2 联动）:
- EvalContext frozen 不可变
- EvalContext 字段默认值（eval_method_hint=None / profile="" / metadata={}）
- EvalContext.metadata 默认值不共享（每次创建新 dict）
- EvalContext 接受 L2 EvolutionArtifact + EvalTask（跨模块类型联动）
- EvalBridge runtime_checkable（mock 实现通过 isinstance）
- EvalBridge mock 契约验证（evaluate 返 L2 EvalResult / list_available_methods 返 tuple / health_check 返 bool）
- 非实现类不通过 isinstance
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from agent4ml.backend.agents.multiagent.eval.bridge import EvalBridge
from agent4ml.backend.agents.multiagent.eval.types import EvalContext
from agent4ml.backend.agents.multiagent.evolution.promotion_gate import EvalResult, EvalTask


def _make_artifact(version: str = "v1", template_id: str = "default") -> Any:
    """Mock EvolutionArtifact（实现 version/template_id/artifact_hash 属性）."""
    return SimpleNamespace(
        version=version,
        template_id=template_id,
        artifact_hash=f"hash_{version}_{template_id}",
    )


def _make_task(task_id: str = "t1") -> EvalTask:
    return EvalTask(task_id=task_id, goal="g", success_criteria="sc")


# ── EvalContext 测试 ──────────────────────────────────────


class TestEvalContext:
    def test_frozen_immutable(self):
        ctx = EvalContext(
            candidate=_make_artifact(),
            baseline=_make_artifact(),
            task_sample=(_make_task(),),
        )
        assert is_dataclass(ctx)
        with pytest.raises(FrozenInstanceError):
            ctx.profile = "codex"  # type: ignore[misc]

    def test_defaults(self):
        ctx = EvalContext(
            candidate=_make_artifact(),
            baseline=_make_artifact(),
            task_sample=(),
        )
        assert ctx.eval_method_hint is None
        assert ctx.profile == ""
        assert ctx.metadata == {}

    def test_metadata_default_not_shared(self):
        ctx1 = EvalContext(
            candidate=_make_artifact(),
            baseline=_make_artifact(),
            task_sample=(),
        )
        ctx2 = EvalContext(
            candidate=_make_artifact(),
            baseline=_make_artifact(),
            task_sample=(),
        )
        ctx1.metadata["key"] = "value"
        assert "key" not in ctx2.metadata

    def test_accepts_l2_types(self):
        """EvalContext 接受 L2 EvolutionArtifact + EvalTask（跨模块类型联动）."""
        ctx = EvalContext(
            candidate=_make_artifact("v2", "ctx_sum"),
            baseline=_make_artifact("v1", "ctx_sum"),
            task_sample=(_make_task("t1"), _make_task("t2")),
            eval_method_hint="longitudinal_pairs",
            profile="codex",
            metadata={"task_type": "open_ended"},
        )
        assert ctx.candidate.version == "v2"
        assert ctx.baseline.version == "v1"
        assert len(ctx.task_sample) == 2
        assert ctx.task_sample[0].task_id == "t1"
        assert ctx.eval_method_hint == "longitudinal_pairs"
        assert ctx.profile == "codex"
        assert ctx.metadata["task_type"] == "open_ended"


# ── EvalBridge Protocol 测试 ──────────────────────────────


class _MockBridge:
    """Mock EvalBridge 实现（验证 runtime_checkable + 契约）."""

    def evaluate(self, ctx: EvalContext) -> EvalResult:
        return EvalResult(
            candidate_score=0.8,
            baseline_score=0.5,
            ci_low=0.6,
            ci_high=0.9,
            sample_size=10,
            method_used="programmatic",
            success=True,
        )

    def list_available_methods(self) -> tuple[str, ...]:
        return ("programmatic", "llm_judge", "longitudinal_pairs")

    def health_check(self) -> bool:
        return True


class _IncompleteBridge:
    """缺少 list_available_methods 的不完整实现（不应通过 isinstance）."""

    def evaluate(self, ctx: EvalContext) -> EvalResult: ...
    def health_check(self) -> bool: ...


class TestEvalBridgeProtocol:
    def test_runtime_checkable_isinstance(self):
        bridge = _MockBridge()
        assert isinstance(bridge, EvalBridge)

    def test_runtime_checkable_rejects_incomplete(self):
        incomplete = _IncompleteBridge()
        assert not isinstance(incomplete, EvalBridge)

    def test_runtime_checkable_rejects_plain_object(self):
        assert not isinstance(object(), EvalBridge)
        assert not isinstance("not_a_bridge", EvalBridge)

    def test_mock_evaluate_returns_l2_eval_result(self):
        """mock evaluate 返 L2 EvalResult（跨模块类型联动）."""
        bridge = _MockBridge()
        ctx = EvalContext(
            candidate=_make_artifact(),
            baseline=_make_artifact(),
            task_sample=(_make_task(),),
        )
        result = bridge.evaluate(ctx)
        assert isinstance(result, EvalResult)
        assert result.candidate_score == 0.8
        assert result.success is True
        assert result.method_used == "programmatic"

    def test_mock_list_available_methods_returns_tuple(self):
        bridge = _MockBridge()
        methods = bridge.list_available_methods()
        assert isinstance(methods, tuple)
        assert "programmatic" in methods

    def test_mock_health_check_returns_bool(self):
        bridge = _MockBridge()
        assert isinstance(bridge.health_check(), bool)
        assert bridge.health_check() is True
