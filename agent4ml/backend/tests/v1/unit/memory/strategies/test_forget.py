from __future__ import annotations

import pytest
from dataclasses import replace

from agent4ml.backend.agents.memory.config import get_memory_config, set_memory_config
from agent4ml.backend.agents.memory.schema import MemoryTrace, MemoryType
from agent4ml.backend.agents.memory.strategies.default._constants import (
    DECAY_PARAMS,
    FORGET_THRESHOLDS,
)
from agent4ml.backend.agents.memory.strategies.default.forget import CompositeForgetPolicy


@pytest.fixture(autouse=True)
def _reset_config() -> None:
    original = get_memory_config()
    yield
    set_memory_config(original)


def _make_trace(
    *,
    type: MemoryType = MemoryType.EPISODIC,
    last_accessed: float = 1000.0,
    created_at: float = 1000.0,
    access_count: int = 0,
    importance: float = 0.5,
) -> MemoryTrace:
    params = DECAY_PARAMS[type.value]
    return MemoryTrace(
        id="t1",
        content="c",
        type=type,
        strength=params["base_strength"],
        base_strength=params["base_strength"],
        decay_rate=params["decay_rate"],
        access_count=access_count,
        last_accessed=last_accessed,
        importance=importance,
        created_at=created_at,
    )


class TestTtlRule:
    def test_ttl_expired_returns_true(self) -> None:
        """TTL 过期：now - last_accessed > ttl_hours×3600 → True。"""
        trace = _make_trace(last_accessed=1000.0)
        policy = CompositeForgetPolicy()
        # ttl_hours=720 → 720×3600=2592000 秒
        now = 1000.0 + 2592000 + 1  # 超 1 秒
        assert policy.should_forget(trace, now) is True

    def test_ttl_not_expired_returns_false(self) -> None:
        """TTL 未过期 + strength 足够 → False。"""
        trace = _make_trace(last_accessed=1000.0, access_count=10)
        policy = CompositeForgetPolicy()
        now = 1000.0 + 3600  # 1h 后，远未到 TTL
        # strength 应该 > threshold（base_strength=0.7 + 强化）
        assert policy.should_forget(trace, now) is False

    def test_unaccessed_uses_created_at(self) -> None:
        """last_accessed<=0 时用 created_at 算 TTL。"""
        trace = _make_trace(created_at=1000.0, last_accessed=0.0)
        policy = CompositeForgetPolicy()
        now = 1000.0 + 2592000 + 1
        assert policy.should_forget(trace, now) is True


class TestStrengthRule:
    def test_strength_below_threshold_returns_true(self) -> None:
        """strength 低于阈值 → True（衰减到阈值下）。"""
        trace = _make_trace(last_accessed=1000.0, access_count=0, importance=0.0)
        policy = CompositeForgetPolicy()
        # time_hours 极大，衰减项→0，strength 接近 0 < threshold(0.1)
        now = 1000.0 + 3600 * 100000
        assert policy.should_forget(trace, now) is True

    def test_strength_above_threshold_returns_false(self) -> None:
        """strength 高 + TTL 未过期 → False。"""
        trace = _make_trace(last_accessed=1000.0, access_count=20, importance=1.0)
        policy = CompositeForgetPolicy()
        now = 1000.0  # time_hours=0，strength 高
        assert policy.should_forget(trace, now) is False


class TestNoResolveConflict:
    def test_no_resolve_conflict_method(self) -> None:
        """B3：CompositeForgetPolicy 无 resolve_conflict 方法。"""
        policy = CompositeForgetPolicy()
        assert not hasattr(policy, "resolve_conflict")

    def test_class_no_resolve_conflict(self) -> None:
        """B3：类定义无 resolve_conflict。"""
        assert not hasattr(CompositeForgetPolicy, "resolve_conflict")


class TestRuntimeConfigOverride:
    def test_config_forget_overrides_thresholds(self) -> None:
        """set_memory_config 后用 config.forget 覆盖阈值。"""
        trace = _make_trace(last_accessed=1000.0, access_count=10)
        policy = CompositeForgetPolicy()
        now = 1000.0 + 3600  # 1h 后

        # 默认 ttl_hours=720，1h 未过期 → False
        assert policy.should_forget(trace, now) is False

        # 覆盖 config：ttl_hours=0.5（半小时即过期）
        new_config = replace(
            get_memory_config(),
            forget={"strength_threshold": 0.1, "ttl_hours": 0.5},
        )
        set_memory_config(new_config)
        # 1h > 0.5h → TTL 过期 → True
        assert policy.should_forget(trace, now) is True

    def test_config_strength_threshold_override(self) -> None:
        """config.forget.strength_threshold 覆盖。"""
        trace = _make_trace(last_accessed=1000.0, access_count=10, importance=1.0)
        policy = CompositeForgetPolicy()
        now = 1000.0  # strength 高

        # 默认 threshold=0.1，strength 高 → False
        assert policy.should_forget(trace, now) is False

        # 覆盖 threshold=0.99（极高，正常 strength 都 < 0.99）
        new_config = replace(
            get_memory_config(),
            forget={"strength_threshold": 0.99, "ttl_hours": 720},
        )
        set_memory_config(new_config)
        # strength < 0.99 → True
        assert policy.should_forget(trace, now) is True


class TestDecayPolicyDependency:
    def test_uses_injected_decay_policy(self) -> None:
        """规则 2 调 decay_policy.compute_strength。"""
        from agent4ml.backend.agents.memory.strategies.default.decay import EbbinghausDecayPolicy

        class _SpyDecay(EbbinghausDecayPolicy):
            def __init__(self) -> None:
                super().__init__()
                self.called = False

            def compute_strength(self, trace: MemoryTrace, now: float) -> float:
                self.called = True
                return 0.5  # > threshold，不触发规则 2

        spy = _SpyDecay()
        policy = CompositeForgetPolicy(decay_policy=spy)
        trace = _make_trace(last_accessed=1000.0)
        policy.should_forget(trace, 1000.0)  # TTL 未过期 → 走规则 2
        assert spy.called is True

    def test_default_decay_policy_when_none(self) -> None:
        """decay_policy=None 时内部构造 EbbinghausDecayPolicy。"""
        from agent4ml.backend.agents.memory.strategies.default.decay import EbbinghausDecayPolicy

        policy = CompositeForgetPolicy(decay_policy=None)
        assert isinstance(policy._decay_policy, EbbinghausDecayPolicy)
