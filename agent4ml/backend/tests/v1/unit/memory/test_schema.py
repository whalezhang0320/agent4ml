from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from agent4ml.backend.agents.memory.schema import (
    Association,
    MemoryTrace,
    MemoryType,
    OperationLog,
)


def _make_trace(**overrides: Any) -> MemoryTrace:
    defaults: dict[str, Any] = {
        "id": "t1",
        "content": "hello",
        "type": MemoryType.EPISODIC,
    }
    defaults.update(overrides)
    return MemoryTrace(**defaults)


class TestMemoryType:
    def test_episodic_value(self) -> None:
        assert MemoryType.EPISODIC.value == "episodic"

    def test_semantic_value(self) -> None:
        assert MemoryType.SEMANTIC.value == "semantic"

    def test_procedural_value(self) -> None:
        assert MemoryType.PROCEDURAL.value == "procedural"

    def test_is_str_enum(self) -> None:
        assert isinstance(MemoryType.EPISODIC, str)
        assert MemoryType.EPISODIC == "episodic"


class TestAssociation:
    def test_default_strength_and_type(self) -> None:
        assoc = Association(target_id="t2")
        assert assoc.strength == 0.5
        assert assoc.type == "related"

    def test_custom_strength_and_type(self) -> None:
        assoc = Association(target_id="t2", strength=0.9, type="causal")
        assert assoc.strength == 0.9
        assert assoc.type == "causal"

    def test_frozen(self) -> None:
        assoc = Association(target_id="t2")
        with pytest.raises(dataclasses.FrozenInstanceError):
            assoc.strength = 0.1


class TestOperationLog:
    def test_default_actor_and_diff_none(self) -> None:
        log = OperationLog(timestamp=1.0, operation="encode")
        assert log.actor is None
        assert log.diff is None

    def test_custom_actor_and_diff(self) -> None:
        log = OperationLog(
            timestamp=1.0,
            operation="reconsolidate",
            actor="turn-5",
            diff={"content": ("old", "new")},
        )
        assert log.actor == "turn-5"
        assert log.diff == {"content": ("old", "new")}

    def test_frozen(self) -> None:
        log = OperationLog(timestamp=1.0, operation="encode")
        with pytest.raises(dataclasses.FrozenInstanceError):
            log.actor = "x"


class TestMemoryTraceDefaults:
    def test_required_fields(self) -> None:
        trace = _make_trace()
        assert trace.id == "t1"
        assert trace.content == "hello"
        assert trace.type is MemoryType.EPISODIC

    def test_default_strength_zero(self) -> None:
        assert _make_trace().strength == 0.0

    def test_default_base_strength(self) -> None:
        assert _make_trace().base_strength == 0.7

    def test_default_decay_rate(self) -> None:
        assert _make_trace().decay_rate == 0.1

    def test_default_access_count_zero(self) -> None:
        assert _make_trace().access_count == 0

    def test_default_importance(self) -> None:
        assert _make_trace().importance == 0.5

    def test_default_associations_empty_tuple(self) -> None:
        assert _make_trace().associations == ()

    def test_default_embedding_none(self) -> None:
        assert _make_trace().embedding is None

    def test_default_source_none(self) -> None:
        assert _make_trace().source is None

    def test_default_metadata_empty_dict(self) -> None:
        assert _make_trace().metadata == {}

    def test_default_operation_log_empty_tuple(self) -> None:
        assert _make_trace().operation_log == ()

    def test_frozen(self) -> None:
        trace = _make_trace()
        with pytest.raises(dataclasses.FrozenInstanceError):
            trace.strength = 0.5


class TestWithStrength:
    def test_returns_new_instance(self) -> None:
        trace = _make_trace()
        new_trace = trace.with_strength(0.8, 1000.0)
        assert new_trace is not trace

    def test_original_unchanged(self) -> None:
        trace = _make_trace()
        trace.with_strength(0.8, 1000.0)
        assert trace.strength == 0.0
        assert trace.access_count == 0
        assert trace.last_accessed == 0.0

    def test_new_strength_set(self) -> None:
        new_trace = _make_trace().with_strength(0.8, 1000.0)
        assert new_trace.strength == 0.8

    def test_access_count_incremented(self) -> None:
        trace = _make_trace()
        new_trace = trace.with_strength(0.8, 1000.0)
        assert new_trace.access_count == 1
        new_trace2 = new_trace.with_strength(0.9, 2000.0)
        assert new_trace2.access_count == 2

    def test_last_accessed_updated(self) -> None:
        new_trace = _make_trace().with_strength(0.8, 1000.0)
        assert new_trace.last_accessed == 1000.0

    def test_does_not_append_operation_log(self) -> None:
        trace = _make_trace()
        new_trace = trace.with_strength(0.8, 1000.0)
        assert new_trace.operation_log == trace.operation_log == ()


class TestWithOperation:
    def test_returns_new_instance(self) -> None:
        trace = _make_trace()
        log = OperationLog(timestamp=1.0, operation="encode")
        new_trace = trace.with_operation(log)
        assert new_trace is not trace

    def test_original_unchanged(self) -> None:
        trace = _make_trace()
        log = OperationLog(timestamp=1.0, operation="encode")
        trace.with_operation(log)
        assert trace.operation_log == ()

    def test_appends_one_log(self) -> None:
        trace = _make_trace()
        log = OperationLog(timestamp=1.0, operation="encode")
        new_trace = trace.with_operation(log)
        assert len(new_trace.operation_log) == 1
        assert new_trace.operation_log[0] is log

    def test_appends_multiple_logs_in_order(self) -> None:
        trace = _make_trace()
        log1 = OperationLog(timestamp=1.0, operation="encode")
        log2 = OperationLog(timestamp=2.0, operation="associate")
        new_trace = trace.with_operation(log1).with_operation(log2)
        assert len(new_trace.operation_log) == 2
        assert new_trace.operation_log[0] is log1
        assert new_trace.operation_log[1] is log2

    def test_fifo_drops_oldest_at_max_log_20(self) -> None:
        trace = _make_trace()
        for i in range(20):
            trace = trace.with_operation(
                OperationLog(timestamp=float(i), operation="encode")
            )
        assert len(trace.operation_log) == 20
        assert trace.operation_log[0].timestamp == 0.0

        # 第 21 条 → 丢最老（timestamp=0.0）
        trace = trace.with_operation(
            OperationLog(timestamp=20.0, operation="forget")
        )
        assert len(trace.operation_log) == 20
        assert trace.operation_log[0].timestamp == 1.0
        assert trace.operation_log[-1].timestamp == 20.0

    def test_custom_max_log(self) -> None:
        trace = _make_trace()
        for i in range(3):
            trace = trace.with_operation(
                OperationLog(timestamp=float(i), operation="encode"),
                max_log=2,
            )
        assert len(trace.operation_log) == 2
        assert trace.operation_log[0].timestamp == 1.0
        assert trace.operation_log[-1].timestamp == 2.0
