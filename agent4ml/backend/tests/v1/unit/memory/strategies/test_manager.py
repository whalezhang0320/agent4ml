from __future__ import annotations

import hashlib

import pytest
from dataclasses import replace

from agent4ml.backend.agents.memory.config import get_memory_config, set_memory_config
from agent4ml.backend.agents.memory.exceptions import MemoryNotFoundError
from agent4ml.backend.agents.memory.schema import Association, MemoryTrace, MemoryType
from agent4ml.backend.agents.memory.strategies.default._constants import (
    ASSOCIATE_DEFAULTS,
    DECAY_PARAMS,
)
from agent4ml.backend.agents.memory.strategies.default.manager import (
    DefaultMemoryManager,
    set_turn_id,
)


@pytest.fixture(autouse=True)
def _reset_turn_id() -> None:
    set_turn_id(None)
    yield
    set_turn_id(None)


@pytest.fixture(autouse=True)
def _reset_config() -> None:
    original = get_memory_config()
    yield
    set_memory_config(original)


class _MockStore:
    """mock MemoryStore：dict-based 记录 add/update/batch_update 调用。"""

    def __init__(self) -> None:
        self._data: dict[str, MemoryTrace] = {}
        self.add_calls: list[MemoryTrace] = []
        self.update_calls: list[MemoryTrace] = []
        self.batch_update_calls: list[list[MemoryTrace]] = []

    def add(self, trace: MemoryTrace) -> None:
        self.add_calls.append(trace)
        self._data[trace.id] = trace

    def get(self, trace_id: str) -> MemoryTrace | None:
        return self._data.get(trace_id)

    def update(self, trace: MemoryTrace) -> None:
        self.update_calls.append(trace)
        self._data[trace.id] = trace

    def batch_update(self, traces: list[MemoryTrace]) -> None:
        self.batch_update_calls.append(traces)
        for t in traces:
            self._data[t.id] = t

    def remove(self, trace_id: str) -> None:
        self._data.pop(trace_id, None)

    def list_by_type(self, type: MemoryType) -> list[MemoryTrace]:
        return [t for t in self._data.values() if t.type == type]

    def list_all(self) -> list[MemoryTrace]:
        return list(self._data.values())


class TestComputeTraceId:
    def test_id_is_sha256_16(self) -> None:
        """F2：id = SHA256(content+type.value)[:16]。"""
        content = "hello"
        type_ = MemoryType.EPISODIC
        expected = hashlib.sha256(f"{content}\x00{type_.value}".encode("utf-8")).hexdigest()[:16]
        assert DefaultMemoryManager._compute_trace_id(content, type_) == expected

    def test_same_content_same_type_same_id(self) -> None:
        """同内容同 type → 同 id（去重基础）。"""
        id1 = DefaultMemoryManager._compute_trace_id("hello", MemoryType.EPISODIC)
        id2 = DefaultMemoryManager._compute_trace_id("hello", MemoryType.EPISODIC)
        assert id1 == id2

    def test_different_type_different_id(self) -> None:
        """同内容不同 type → 不同 id。"""
        id_e = DefaultMemoryManager._compute_trace_id("hello", MemoryType.EPISODIC)
        id_s = DefaultMemoryManager._compute_trace_id("hello", MemoryType.SEMANTIC)
        assert id_e != id_s


class TestAddAssociationWithLru:
    def _make_trace(self, assocs: tuple[Association, ...] = ()) -> MemoryTrace:
        return MemoryTrace(id="t", content="c", type=MemoryType.EPISODIC, associations=assocs)

    def test_under_max_appends(self) -> None:
        trace = self._make_trace(())
        new = Association(target_id="b", strength=0.5)
        result = DefaultMemoryManager._add_association_with_lru(trace, new, max_assocs=20)
        assert len(result.associations) == 1
        assert result.associations[0] is new
        # 原始 trace 不变
        assert len(trace.associations) == 0

    def test_over_max_evicts_weakest(self) -> None:
        """D3：超 max 时淘汰 strength 最低的。"""
        existing = (
            Association(target_id="a", strength=0.3),
            Association(target_id="b", strength=0.8),
            Association(target_id="c", strength=0.1),  # 最弱
        )
        trace = self._make_trace(existing)
        new = Association(target_id="d", strength=0.5)
        result = DefaultMemoryManager._add_association_with_lru(trace, new, max_assocs=3)
        assert len(result.associations) == 3
        target_ids = [a.target_id for a in result.associations]
        # 最弱的 c 被淘汰
        assert "c" not in target_ids
        assert "d" in target_ids

    def test_at_max_keeps_strongest_minus_one_plus_new(self) -> None:
        """正好 max 时淘汰最弱，保留 max-1 最强 + 新关联。"""
        existing = tuple(Association(target_id=str(i), strength=float(i) / 10) for i in range(5))
        trace = self._make_trace(existing)
        new = Association(target_id="new", strength=0.9)
        result = DefaultMemoryManager._add_association_with_lru(trace, new, max_assocs=5)
        assert len(result.associations) == 5
        # 最弱的 target_id="0" (strength=0.0) 被淘汰
        assert "0" not in [a.target_id for a in result.associations]


class TestGetActor:
    def test_default_none(self) -> None:
        """C：未设 ContextVar 返 None。"""
        store = _MockStore()
        manager = DefaultMemoryManager(store)
        assert manager._get_actor() is None

    def test_set_turn_id_returns_value(self) -> None:
        """C：set_turn_id 后返 turn_id。"""
        set_turn_id("turn-5")
        store = _MockStore()
        manager = DefaultMemoryManager(store)
        assert manager._get_actor() == "turn-5"


class TestEmitJournal:
    def test_journal_none_no_emit(self) -> None:
        """B：journal=None 时不 emit。"""
        store = _MockStore()
        manager = DefaultMemoryManager(store, journal=None)
        # 不抛错即可
        manager._emit_journal("memory.test", {"k": "v"})

    def test_journal_mock_called(self) -> None:
        """B：journal=mock 时被调用。"""
        events: list[tuple[str, dict]] = []

        def journal(event: str, payload: dict) -> None:
            events.append((event, payload))

        store = _MockStore()
        manager = DefaultMemoryManager(store, journal=journal)
        manager._emit_journal("memory.test", {"k": "v"})
        assert len(events) == 1
        assert events[0] == ("memory.test", {"k": "v"})


class TestEncode:
    def test_id_is_sha256_16(self) -> None:
        """F2：encode 生成的 id = SHA256[:16]。"""
        store = _MockStore()
        manager = DefaultMemoryManager(store)
        trace = manager.encode("hello", MemoryType.EPISODIC)
        expected = DefaultMemoryManager._compute_trace_id("hello", MemoryType.EPISODIC)
        assert trace.id == expected

    def test_strength_equals_base_strength(self) -> None:
        """A1：encode strength = base_strength。"""
        store = _MockStore()
        manager = DefaultMemoryManager(store)
        trace = manager.encode("hello", MemoryType.EPISODIC)
        assert trace.strength == DECAY_PARAMS["episodic"]["base_strength"]
        assert trace.base_strength == DECAY_PARAMS["episodic"]["base_strength"]

    def test_store_add_called_once(self) -> None:
        store = _MockStore()
        manager = DefaultMemoryManager(store)
        manager.encode("hello", MemoryType.EPISODIC)
        assert len(store.add_calls) == 1

    def test_operation_log_contains_encode(self) -> None:
        """traceability A：operation_log 含 encode 条目。"""
        store = _MockStore()
        manager = DefaultMemoryManager(store)
        trace = manager.encode("hello", MemoryType.EPISODIC)
        assert len(trace.operation_log) == 1
        assert trace.operation_log[0].operation == "encode"

    def test_journal_emit_memory_encode(self) -> None:
        """traceability B：emit memory.encode 事件。"""
        events: list[tuple[str, dict]] = []

        def journal(event: str, payload: dict) -> None:
            events.append((event, payload))

        store = _MockStore()
        manager = DefaultMemoryManager(store, journal=journal)
        manager.encode("hello", MemoryType.EPISODIC)
        assert len(events) == 1
        assert events[0][0] == "memory.encode"
        assert events[0][1]["type"] == "episodic"

    def test_idempotent_same_content_returns_existing(self) -> None:
        """点4：同 content+type 第二次 encode 返回旧 trace，不调 store.add。"""
        store = _MockStore()
        manager = DefaultMemoryManager(store)
        trace1 = manager.encode("hello", MemoryType.EPISODIC)
        trace2 = manager.encode("hello", MemoryType.EPISODIC)
        assert trace1 is trace2
        assert len(store.add_calls) == 1  # 只 add 一次

    def test_idempotent_emits_duplicate_event(self) -> None:
        """点4：幂等时 emit memory.encode.duplicate 事件。"""
        events: list[str] = []

        def journal(event: str, payload: dict) -> None:
            events.append(event)

        store = _MockStore()
        manager = DefaultMemoryManager(store, journal=journal)
        manager.encode("hello", MemoryType.EPISODIC)
        manager.encode("hello", MemoryType.EPISODIC)  # 幂等
        assert events == ["memory.encode", "memory.encode.duplicate"]

    def test_three_types_different_base_strength(self) -> None:
        """三类记忆 base_strength 不同。"""
        store = _MockStore()
        manager = DefaultMemoryManager(store)
        e = manager.encode("e", MemoryType.EPISODIC)
        s = manager.encode("s", MemoryType.SEMANTIC)
        p = manager.encode("p", MemoryType.PROCEDURAL)
        assert e.base_strength == 0.7
        assert s.base_strength == 0.8
        assert p.base_strength == 0.9

    def test_metadata_default_empty_dict(self) -> None:
        store = _MockStore()
        manager = DefaultMemoryManager(store)
        trace = manager.encode("hello", MemoryType.EPISODIC)
        assert trace.metadata == {}

    def test_metadata_custom(self) -> None:
        store = _MockStore()
        manager = DefaultMemoryManager(store)
        trace = manager.encode("hello", MemoryType.EPISODIC, metadata={"tag": "travel"})
        assert trace.metadata == {"tag": "travel"}

    def test_source_passed_through(self) -> None:
        store = _MockStore()
        manager = DefaultMemoryManager(store)
        trace = manager.encode("hello", MemoryType.EPISODIC, source="thread-123")
        assert trace.source == "thread-123"

    def test_importance_passed_through(self) -> None:
        store = _MockStore()
        manager = DefaultMemoryManager(store)
        trace = manager.encode("hello", MemoryType.EPISODIC, importance=0.9)
        assert trace.importance == 0.9

    def test_actor_from_turn_id(self) -> None:
        """traceability C：actor 从 ContextVar 取。"""
        set_turn_id("turn-abc")
        store = _MockStore()
        manager = DefaultMemoryManager(store)
        trace = manager.encode("hello", MemoryType.EPISODIC)
        assert trace.operation_log[0].actor == "turn-abc"


class TestAssociate:
    def _setup_two_traces(self, store: _MockStore) -> tuple[str, str]:
        """预先 encode 两条 trace，返回 id。"""
        manager = DefaultMemoryManager(store)
        t1 = manager.encode("content A", MemoryType.EPISODIC)
        t2 = manager.encode("content B", MemoryType.SEMANTIC)
        return t1.id, t2.id

    def test_bidirectional_association(self) -> None:
        """双向加 Association：A→B + B→A。"""
        store = _MockStore()
        manager = DefaultMemoryManager(store)
        id_a, id_b = self._setup_two_traces(store)

        manager.associate(id_a, id_b)

        updated_a = store.get(id_a)
        updated_b = store.get(id_b)
        assert any(a.target_id == id_b for a in updated_a.associations)
        assert any(a.target_id == id_a for a in updated_b.associations)

    def test_store_update_called_twice(self) -> None:
        """双向 update：store.update 调用两次。"""
        store = _MockStore()
        manager = DefaultMemoryManager(store)
        id_a, id_b = self._setup_two_traces(store)
        store.update_calls.clear()  # 清除 encode 阶段的 update

        manager.associate(id_a, id_b)
        assert len(store.update_calls) == 2

    def test_operation_log_contains_associate(self) -> None:
        """traceability A：operation_log 含 associate 条目。"""
        store = _MockStore()
        manager = DefaultMemoryManager(store)
        id_a, id_b = self._setup_two_traces(store)

        manager.associate(id_a, id_b)

        updated_a = store.get(id_a)
        last_log = updated_a.operation_log[-1]
        assert last_log.operation == "associate"
        assert last_log.diff["target"] == id_b

    def test_journal_emit_memory_associate(self) -> None:
        """traceability B：emit memory.associate 事件。"""
        events: list[str] = []

        def journal(event: str, payload: dict) -> None:
            events.append(event)

        store = _MockStore()
        manager = DefaultMemoryManager(store, journal=journal)
        id_a, id_b = self._setup_two_traces(store)
        events.clear()  # 清除 encode 阶段事件

        manager.associate(id_a, id_b)
        assert events == ["memory.associate"]

    def test_lru_evicts_weakest_when_over_max(self) -> None:
        """D3：超 max_associations_per_trace 时淘汰 strength 最低。"""
        store = _MockStore()
        manager = DefaultMemoryManager(store)
        id_a = manager.encode("main", MemoryType.EPISODIC).id

        # 给 id_a 加 max 条关联（用其他 trace）
        max_assocs = ASSOCIATE_DEFAULTS["max_associations_per_trace"]
        other_ids = []
        for i in range(max_assocs):
            other_id = manager.encode(f"other-{i}", MemoryType.EPISODIC).id
            other_ids.append(other_id)
            manager.associate(id_a, other_id, strength=float(i) / max_assocs)

        # 此时 id_a 已有 max 条关联，再加一条应淘汰最弱的（strength=0.0 的 other-0）
        new_id = manager.encode("new-assoc", MemoryType.EPISODIC).id
        manager.associate(id_a, new_id, strength=0.5)

        updated_a = store.get(id_a)
        # 总数仍 = max
        assert len(updated_a.associations) == max_assocs
        # 最弱的 other-0（strength=0.0）被淘汰
        target_ids = [a.target_id for a in updated_a.associations]
        assert other_ids[0] not in target_ids
        assert new_id in target_ids

    def test_trace_not_found_raises(self) -> None:
        """trace 不存在抛 MemoryNotFoundError。"""
        store = _MockStore()
        manager = DefaultMemoryManager(store)
        id_a = manager.encode("real", MemoryType.EPISODIC).id

        with pytest.raises(MemoryNotFoundError):
            manager.associate(id_a, "nonexistent-id")

    def test_trace_a_not_found_raises(self) -> None:
        """trace_id_a 不存在抛 MemoryNotFoundError。"""
        store = _MockStore()
        manager = DefaultMemoryManager(store)
        id_b = manager.encode("real-b", MemoryType.EPISODIC).id

        with pytest.raises(MemoryNotFoundError):
            manager.associate("nonexistent-a", id_b)

    def test_custom_strength_and_type(self) -> None:
        """自定义 strength + type 传入。"""
        store = _MockStore()
        manager = DefaultMemoryManager(store)
        id_a, id_b = self._setup_two_traces(store)

        manager.associate(id_a, id_b, strength=0.9, type="causal")

        updated_a = store.get(id_a)
        assoc = next(a for a in updated_a.associations if a.target_id == id_b)
        assert assoc.strength == 0.9
        assert assoc.type == "causal"

    def test_default_strength_and_type(self) -> None:
        """默认 strength=0.5 + type=related。"""
        store = _MockStore()
        manager = DefaultMemoryManager(store)
        id_a, id_b = self._setup_two_traces(store)

        manager.associate(id_a, id_b)

        updated_a = store.get(id_a)
        assoc = next(a for a in updated_a.associations if a.target_id == id_b)
        assert assoc.strength == ASSOCIATE_DEFAULTS["default_strength"]
        assert assoc.type == ASSOCIATE_DEFAULTS["default_type"]


class TestConsolidate:
    def _setup_traces(self, store: _MockStore, n: int = 2) -> list[str]:
        manager = DefaultMemoryManager(store)
        ids = []
        for i in range(n):
            t = manager.encode(f"content-{i}", MemoryType.EPISODIC)
            ids.append(t.id)
        return ids

    def test_min_traces_validation(self) -> None:
        """E1：少于 min 抛 ValueError。"""
        store = _MockStore()
        manager = DefaultMemoryManager(store)
        with pytest.raises(ValueError, match="at least"):
            manager.consolidate([], "merged")

    def test_max_traces_validation(self) -> None:
        """E1：超过 max 抛 ValueError。"""
        store = _MockStore()
        manager = DefaultMemoryManager(store)
        ids = self._setup_traces(store, n=11)
        with pytest.raises(ValueError, match="at most"):
            manager.consolidate(ids, "merged")

    def test_creates_semantic_trace(self) -> None:
        """创建 semantic trace。"""
        store = _MockStore()
        manager = DefaultMemoryManager(store)
        ids = self._setup_traces(store, n=2)

        result = manager.consolidate(ids, "merged content")
        assert result.type is MemoryType.SEMANTIC
        assert result.content == "merged content"

    def test_old_traces_marked_forgotten(self) -> None:
        """C1：旧 trace 标记 forgotten，不删除。"""
        store = _MockStore()
        manager = DefaultMemoryManager(store)
        ids = self._setup_traces(store, n=2)

        new_trace = manager.consolidate(ids, "merged content")

        for old_id in ids:
            old_trace = store.get(old_id)
            assert old_trace is not None  # 未删除
            assert old_trace.metadata["forgotten"] is True
            assert old_trace.metadata["consolidated_into"] == new_trace.id

    def test_consolidated_from_metadata(self) -> None:
        """新 trace metadata.consolidated_from = trace_ids。"""
        store = _MockStore()
        manager = DefaultMemoryManager(store)
        ids = self._setup_traces(store, n=3)

        new_trace = manager.consolidate(ids, "merged")
        assert new_trace.metadata["consolidated_from"] == ids

    def test_new_trace_operation_log_contains_consolidate(self) -> None:
        """traceability A：新 trace operation_log 含 consolidate 条目。"""
        store = _MockStore()
        manager = DefaultMemoryManager(store)
        ids = self._setup_traces(store, n=2)

        new_trace = manager.consolidate(ids, "merged")
        last_log = new_trace.operation_log[-1]
        assert last_log.operation == "consolidate"

    def test_old_trace_operation_log_contains_forget(self) -> None:
        """traceability A：旧 trace operation_log 含 forget 条目。"""
        store = _MockStore()
        manager = DefaultMemoryManager(store)
        ids = self._setup_traces(store, n=2)

        manager.consolidate(ids, "merged")

        for old_id in ids:
            old_trace = store.get(old_id)
            last_log = old_trace.operation_log[-1]
            assert last_log.operation == "forget"

    def test_batch_update_called_once(self) -> None:
        """F2：batch_update 调用一次（非 N 次 update）。"""
        store = _MockStore()
        manager = DefaultMemoryManager(store)
        ids = self._setup_traces(store, n=3)

        manager.consolidate(ids, "merged")
        assert len(store.batch_update_calls) == 1
        assert len(store.batch_update_calls[0]) == 3  # 3 条 forgotten traces

    def test_journal_emit_memory_consolidate(self) -> None:
        """traceability B：emit memory.consolidate 事件。"""
        events: list[str] = []

        def journal(event: str, payload: dict) -> None:
            events.append(event)

        store = _MockStore()
        manager = DefaultMemoryManager(store, journal=journal)
        ids = self._setup_traces(store, n=2)
        events.clear()

        manager.consolidate(ids, "merged")
        assert events == ["memory.consolidate"]

    def test_merged_content_stored_directly(self) -> None:
        """无 LLM：merged_content 直接存入。"""
        store = _MockStore()
        manager = DefaultMemoryManager(store)
        ids = self._setup_traces(store, n=2)

        new_trace = manager.consolidate(ids, "raw merged text from LLM")
        assert new_trace.content == "raw merged text from LLM"

    def test_trace_not_found_raises(self) -> None:
        """trace 不存在抛 MemoryNotFoundError。"""
        store = _MockStore()
        manager = DefaultMemoryManager(store)
        real_id = manager.encode("real", MemoryType.EPISODIC).id

        with pytest.raises(MemoryNotFoundError):
            manager.consolidate([real_id, "nonexistent"], "merged")

    def test_idempotent_same_merged_content(self) -> None:
        """C1：同 merged_content+SEMANTIC 第二次返回旧 semantic trace。"""
        store = _MockStore()
        manager = DefaultMemoryManager(store)
        ids = self._setup_traces(store, n=2)

        trace1 = manager.consolidate(ids, "merged")
        trace2 = manager.consolidate(ids, "merged")  # 幂等
        assert trace1 is trace2

    def test_idempotent_emits_duplicate_event(self) -> None:
        """C1：幂等时 emit memory.consolidate.duplicate 事件。"""
        events: list[str] = []

        def journal(event: str, payload: dict) -> None:
            events.append(event)

        store = _MockStore()
        manager = DefaultMemoryManager(store, journal=journal)
        ids = self._setup_traces(store, n=2)

        manager.consolidate(ids, "merged")
        events.clear()
        manager.consolidate(ids, "merged")  # 幂等
        assert events == ["memory.consolidate.duplicate"]

    def test_importance_boost(self) -> None:
        """新 importance = min(1.0, max(old.importance) + boost)。"""
        store = _MockStore()
        manager = DefaultMemoryManager(store)
        t1 = manager.encode("c1", MemoryType.EPISODIC, importance=0.6)
        t2 = manager.encode("c2", MemoryType.EPISODIC, importance=0.8)

        new_trace = manager.consolidate([t1.id, t2.id], "merged")
        # max(0.6, 0.8) + 0.1 = 0.9
        assert new_trace.importance == 0.9


class TestReconsolidate:
    def test_content_replaced(self) -> None:
        """content 替换为 new_content。"""
        store = _MockStore()
        manager = DefaultMemoryManager(store)
        trace = manager.encode("old content", MemoryType.EPISODIC)

        updated = manager.reconsolidate(trace.id, "new content")
        assert updated.content == "new content"

    def test_strength_preserved(self) -> None:
        """B3：strength 保留原值，不重置。"""
        store = _MockStore()
        manager = DefaultMemoryManager(store)
        trace = manager.encode("old", MemoryType.EPISODIC)
        original_strength = trace.strength

        updated = manager.reconsolidate(trace.id, "new")
        assert updated.strength == original_strength

    def test_last_accessed_updated_to_now(self) -> None:
        """B3：last_accessed=now（视为一次访问）。"""
        store = _MockStore()
        manager = DefaultMemoryManager(store)
        trace = manager.encode("old", MemoryType.EPISODIC)

        updated = manager.reconsolidate(trace.id, "new")
        assert updated.last_accessed >= trace.last_accessed

    def test_id_unchanged(self) -> None:
        """id 不变，content 替换。"""
        store = _MockStore()
        manager = DefaultMemoryManager(store)
        trace = manager.encode("old", MemoryType.EPISODIC)

        updated = manager.reconsolidate(trace.id, "new")
        assert updated.id == trace.id

    def test_reconsolidated_at_metadata(self) -> None:
        """metadata 加 reconsolidated_at=now。"""
        store = _MockStore()
        manager = DefaultMemoryManager(store)
        trace = manager.encode("old", MemoryType.EPISODIC)

        updated = manager.reconsolidate(trace.id, "new")
        assert "reconsolidated_at" in updated.metadata

    def test_operation_log_contains_reconsolidate(self) -> None:
        """traceability A：operation_log 含 reconsolidate 条目。"""
        store = _MockStore()
        manager = DefaultMemoryManager(store)
        trace = manager.encode("old", MemoryType.EPISODIC)

        updated = manager.reconsolidate(trace.id, "new")
        last_log = updated.operation_log[-1]
        assert last_log.operation == "reconsolidate"

    def test_operation_log_records_content_diff(self) -> None:
        """operation_log diff.content 记旧/新前 200 字符。"""
        store = _MockStore()
        manager = DefaultMemoryManager(store)
        trace = manager.encode("old content", MemoryType.EPISODIC)

        updated = manager.reconsolidate(trace.id, "new content")
        last_log = updated.operation_log[-1]
        assert last_log.diff["content"] == ("old content", "new content")

    def test_journal_emit_memory_reconsolidate(self) -> None:
        """traceability B：emit memory.reconsolidate 事件。"""
        events: list[str] = []

        def journal(event: str, payload: dict) -> None:
            events.append(event)

        store = _MockStore()
        manager = DefaultMemoryManager(store, journal=journal)
        trace = manager.encode("old", MemoryType.EPISODIC)
        events.clear()

        manager.reconsolidate(trace.id, "new")
        assert events == ["memory.reconsolidate"]

    def test_trace_not_found_raises(self) -> None:
        """trace 不存在抛 MemoryNotFoundError。"""
        store = _MockStore()
        manager = DefaultMemoryManager(store)

        with pytest.raises(MemoryNotFoundError):
            manager.reconsolidate("nonexistent", "new content")

    def test_original_trace_unchanged(self) -> None:
        """frozen 语义：原 trace 实例不变。"""
        store = _MockStore()
        manager = DefaultMemoryManager(store)
        trace = manager.encode("old", MemoryType.EPISODIC)
        original_content = trace.content

        manager.reconsolidate(trace.id, "new")
        # 原始 trace（store 里被 update 替换，但 encode 返回的引用不变）
        assert trace.content == original_content
