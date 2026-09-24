from __future__ import annotations

import pytest

from agent4ml.backend.agents.state.reducers import (
    CORE_FIELDS,
    merge_memory_recalled,
    merge_memory_updates,
)


class TestMergeMemoryRecalled:
    def test_new_none_preserves_existing(self) -> None:
        existing = [{"id": "1"}]
        assert merge_memory_recalled(existing, None) is existing

    def test_existing_none_returns_new(self) -> None:
        result = merge_memory_recalled(None, [{"id": "1"}])
        assert result == [{"id": "1"}]

    def test_both_none_returns_none(self) -> None:
        assert merge_memory_recalled(None, None) is None

    def test_dedupe_by_id_last_write_wins(self) -> None:
        existing = [{"id": "1", "v": "old"}, {"id": "2"}]
        new = [{"id": "1", "v": "new"}, {"id": "3"}]
        result = merge_memory_recalled(existing, new)
        ids = [item["id"] for item in result]
        assert set(ids) == {"1", "2", "3"}
        item1 = next(item for item in result if item["id"] == "1")
        assert item1["v"] == "new"

    def test_empty_new_preserves_existing(self) -> None:
        existing = [{"id": "1"}]
        assert merge_memory_recalled(existing, []) == [{"id": "1"}]

    def test_empty_existing_returns_new(self) -> None:
        assert merge_memory_recalled([], [{"id": "1"}]) == [{"id": "1"}]

    def test_returns_new_list_not_mutating_existing(self) -> None:
        existing = [{"id": "1"}]
        result = merge_memory_recalled(existing, [{"id": "2"}])
        assert existing == [{"id": "1"}]
        assert len(result) == 2


class TestMergeMemoryUpdates:
    def test_new_none_preserves_existing(self) -> None:
        existing = [{"a": 1}]
        assert merge_memory_updates(existing, None) is existing

    def test_existing_none_returns_new(self) -> None:
        result = merge_memory_updates(None, [{"a": 1}])
        assert result == [{"a": 1}]

    def test_both_none_returns_none(self) -> None:
        assert merge_memory_updates(None, None) is None

    def test_pure_append_no_dedupe(self) -> None:
        existing = [{"a": 1}]
        new = [{"b": 2}, {"a": 1}]
        result = merge_memory_updates(existing, new)
        assert len(result) == 3
        assert result[0] == {"a": 1}
        assert result[1] == {"b": 2}
        assert result[2] == {"a": 1}

    def test_empty_new_preserves_existing(self) -> None:
        existing = [{"a": 1}]
        assert merge_memory_updates(existing, []) == [{"a": 1}]

    def test_empty_existing_returns_new(self) -> None:
        assert merge_memory_updates([], [{"a": 1}]) == [{"a": 1}]


class TestCoreFields:
    def test_contains_recalled_memories(self) -> None:
        assert "recalled_memories" in CORE_FIELDS

    def test_contains_memory_updates(self) -> None:
        assert "memory_updates" in CORE_FIELDS

    def test_still_contains_existing_fields(self) -> None:
        assert "messages" in CORE_FIELDS
        assert "observations" in CORE_FIELDS
        assert "errors" in CORE_FIELDS

    def test_metadata_conflict_detected(self) -> None:
        """metadata 不能含 core field recalled_memories。"""
        from agent4ml.backend.agents.state.reducers import ReducerConflictError, _merge_metadata
        with pytest.raises(ReducerConflictError):
            _merge_metadata({}, {"recalled_memories": []})

    def test_metadata_conflict_memory_updates(self) -> None:
        from agent4ml.backend.agents.state.reducers import ReducerConflictError, _merge_metadata
        with pytest.raises(ReducerConflictError):
            _merge_metadata({}, {"memory_updates": []})
