from __future__ import annotations

import math

import pytest
from dataclasses import replace

from agent4ml.backend.agents.memory.config import get_memory_config, set_memory_config
from agent4ml.backend.agents.memory.schema import MemoryTrace, MemoryType
from agent4ml.backend.agents.memory.strategies.default._constants import (
    DECAY_COEFFICIENTS,
    DECAY_PARAMS,
)
from agent4ml.backend.agents.memory.strategies.default.decay import EbbinghausDecayPolicy


@pytest.fixture(autouse=True)
def _reset_config() -> None:
    original = get_memory_config()
    yield
    set_memory_config(original)


def _make_trace(
    *,
    type: MemoryType = MemoryType.EPISODIC,
    strength: float = 0.0,
    base_strength: float | None = None,
    decay_rate: float | None = None,
    access_count: int = 0,
    last_accessed: float = 0.0,
    importance: float = 0.5,
    created_at: float = 1000.0,
) -> MemoryTrace:
    params = DECAY_PARAMS[type.value]
    return MemoryTrace(
        id="t1",
        content="c",
        type=type,
        strength=strength,
        base_strength=base_strength if base_strength is not None else params["base_strength"],
        decay_rate=decay_rate if decay_rate is not None else params["decay_rate"],
        access_count=access_count,
        last_accessed=last_accessed,
        importance=importance,
        created_at=created_at,
    )


class TestNewMemoryTimeHoursZero:
    def test_new_memory_strength_approx_base_plus_importance(self) -> None:
        """新建记忆 time_hours=0：strength ≈ base_strength + 0 + importance×0.05。"""
        trace = _make_trace(type=MemoryType.EPISODIC, created_at=1000.0, last_accessed=0.0)
        now = 1000.0  # time_hours = 0
        policy = EbbinghausDecayPolicy()
        strength = policy.compute_strength(trace, now)
        expected = DECAY_PARAMS["episodic"]["base_strength"] + 0.0 + 0.5 * DECAY_COEFFICIENTS["importance_boost"]
        assert strength == pytest.approx(expected)


class TestDecay:
    def test_time_hours_increase_strength_decrease(self) -> None:
        """time_hours 增大 → strength 减小（衰减项递减）。"""
        trace = _make_trace(type=MemoryType.EPISODIC, last_accessed=1000.0)
        policy = EbbinghausDecayPolicy()
        s1 = policy.compute_strength(trace, 1000.0)  # time_hours=0
        s2 = policy.compute_strength(trace, 1360.0)  # time_hours=1
        s3 = policy.compute_strength(trace, 10000.0)  # time_hours很大
        assert s1 > s2 > s3

    def test_episodic_decays_fastest(self) -> None:
        """episodic decay_rate=0.1 衰减最快。"""
        trace_e = _make_trace(type=MemoryType.EPISODIC, last_accessed=1000.0)
        trace_s = _make_trace(type=MemoryType.SEMANTIC, last_accessed=1000.0)
        trace_p = _make_trace(type=MemoryType.PROCEDURAL, last_accessed=1000.0)
        policy = EbbinghausDecayPolicy()
        now = 1000.0 + 3600 * 24  # 24h 后
        s_e = policy.compute_strength(trace_e, now)
        s_s = policy.compute_strength(trace_s, now)
        s_p = policy.compute_strength(trace_p, now)
        # procedural 几乎不衰减（最强），semantic 次之，episodic 衰减最快（最弱）
        assert s_p > s_s > s_e


class TestAccessBoost:
    def test_access_count_increase_strength_increase(self) -> None:
        """access_count 增大 → strength 增大（log(1+n) 递增）。"""
        trace_low = _make_trace(access_count=0, last_accessed=1000.0)
        trace_high = _make_trace(access_count=20, last_accessed=1000.0)
        policy = EbbinghausDecayPolicy()
        now = 1000.0
        s_low = policy.compute_strength(trace_low, now)
        s_high = policy.compute_strength(trace_high, now)
        assert s_high > s_low

    def test_access_boost_formula(self) -> None:
        """强化项 = log(1 + access_count) × 0.1。"""
        trace = _make_trace(access_count=2, last_accessed=1000.0)
        policy = EbbinghausDecayPolicy()
        strength = policy.compute_strength(trace, 1000.0)
        expected_boost = math.log(1 + 2) * DECAY_COEFFICIENTS["access_boost"]
        base = DECAY_PARAMS["episodic"]["base_strength"]
        importance_boost = 0.5 * DECAY_COEFFICIENTS["importance_boost"]
        assert strength == pytest.approx(base + expected_boost + importance_boost)


class TestImportance:
    def test_importance_increase_strength_increase(self) -> None:
        """importance 增大 → strength 增大。"""
        trace_low = _make_trace(importance=0.0, last_accessed=1000.0)
        trace_high = _make_trace(importance=1.0, last_accessed=1000.0)
        policy = EbbinghausDecayPolicy()
        now = 1000.0
        s_low = policy.compute_strength(trace_low, now)
        s_high = policy.compute_strength(trace_high, now)
        assert s_high > s_low


class TestClamp:
    def test_strength_in_zero_one_range(self) -> None:
        """strength 永远在 [0.0, 1.0]。"""
        trace = _make_trace(access_count=1000, last_accessed=1000.0, importance=1.0)
        policy = EbbinghausDecayPolicy()
        strength = policy.compute_strength(trace, 1000.0)
        assert 0.0 <= strength <= 1.0

    def test_strength_zero_when_fully_decayed(self) -> None:
        """完全衰减后 strength 不低于 0（衰减项→0，但强化+重要性可能 >0）。"""
        trace = _make_trace(access_count=0, importance=0.0, last_accessed=1000.0)
        policy = EbbinghausDecayPolicy()
        # time_hours 极大，衰减项 → 0
        strength = policy.compute_strength(trace, 1000.0 + 3600 * 100000)
        assert strength >= 0.0


class TestRuntimeConfigOverride:
    def test_config_decay_overrides_constants(self) -> None:
        """set_memory_config 后用 config.decay 覆盖值。"""
        trace = _make_trace(type=MemoryType.EPISODIC, last_accessed=1000.0)
        policy = EbbinghausDecayPolicy()
        # 默认 base_strength=0.7
        s_default = policy.compute_strength(trace, 1000.0)

        # 覆盖 config
        new_config = replace(
            get_memory_config(),
            decay={"episodic": {"base_strength": 0.99, "decay_rate": 0.5}},
        )
        set_memory_config(new_config)
        s_override = policy.compute_strength(trace, 1000.0)
        assert s_override > s_default  # base_strength 0.99 > 0.7


class TestPureFunction:
    def test_does_not_modify_trace(self) -> None:
        """纯函数：多次调用结果一致 + trace 字段不变。"""
        trace = _make_trace(access_count=5, last_accessed=1000.0, importance=0.6)
        policy = EbbinghausDecayPolicy()
        s1 = policy.compute_strength(trace, 2000.0)
        s2 = policy.compute_strength(trace, 2000.0)
        assert s1 == s2
        # trace 字段不变
        assert trace.access_count == 5
        assert trace.last_accessed == 1000.0
        assert trace.importance == 0.6
        assert trace.strength == 0.0  # 原始 strength 字段不被修改

    def test_unaccessed_uses_created_at(self) -> None:
        """last_accessed<=0 时用 created_at 算 time_hours。"""
        trace = _make_trace(created_at=1000.0, last_accessed=0.0)
        policy = EbbinghausDecayPolicy()
        s_now = policy.compute_strength(trace, 1000.0)  # time_hours=0
        s_later = policy.compute_strength(trace, 4600.0)  # time_hours=1
        assert s_now > s_later
