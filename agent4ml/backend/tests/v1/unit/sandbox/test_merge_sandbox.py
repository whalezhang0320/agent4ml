from __future__ import annotations

import pytest

from agent4ml.backend.agents.state.reducers import merge_sandbox
from agent4ml.backend.agents.state.thread_state import create_initial_thread_state


class TestMergeSandboxIdempotent:
    def test_same_id_idempotent(self) -> None:
        existing = {"sandbox_id": "abc"}
        new = {"sandbox_id": "abc"}
        result = merge_sandbox(existing, new)
        assert result == {"sandbox_id": "abc"}

    def test_same_id_returns_existing_object(self) -> None:
        existing = {"sandbox_id": "abc"}
        new = {"sandbox_id": "abc"}
        result = merge_sandbox(existing, new)
        assert result is existing


class TestMergeSandboxFailClosed:
    def test_different_id_raises(self) -> None:
        existing = {"sandbox_id": "abc"}
        new = {"sandbox_id": "xyz"}
        with pytest.raises(ValueError, match="Conflicting sandbox state"):
            merge_sandbox(existing, new)

    def test_existing_id_new_none_raises(self) -> None:
        existing = {"sandbox_id": "abc"}
        new = {"sandbox_id": None}
        with pytest.raises(ValueError):
            merge_sandbox(existing, new)

    def test_none_existing_new_id_raises(self) -> None:
        existing = {"sandbox_id": None}
        new = {"sandbox_id": "abc"}
        with pytest.raises(ValueError):
            merge_sandbox(existing, new)


class TestMergeSandboxNoClear:
    def test_new_none_preserves_existing(self) -> None:
        existing = {"sandbox_id": "abc"}
        result = merge_sandbox(existing, None)
        assert result == {"sandbox_id": "abc"}
        assert result is existing

    def test_new_none_preserves_none(self) -> None:
        result = merge_sandbox(None, None)
        assert result is None


class TestMergeSandboxExistingNone:
    def test_existing_none_new_dict(self) -> None:
        new = {"sandbox_id": "abc"}
        result = merge_sandbox(None, new)
        assert result == {"sandbox_id": "abc"}
        assert result is new

    def test_existing_none_new_none(self) -> None:
        result = merge_sandbox(None, None)
        assert result is None


class TestThreadStateInitial:
    def test_initial_state_has_sandbox_none(self) -> None:
        state = create_initial_thread_state("test question")
        assert state["sandbox"] is None

    def test_initial_state_preserves_existing_fields(self) -> None:
        state = create_initial_thread_state("test")
        assert state["messages"] == []
        assert state["observations"] == []
        assert state["sources"] == []
        assert state["citations"] == []
        assert state["artifacts"] == []
        assert state["reflection_items"] == []
        assert state["errors"] == []
        assert state["metadata"] == {}
        assert state["governance"] is None
        assert state["user_input"] == "test"

    def test_initial_state_field_count(self) -> None:
        state = create_initial_thread_state("x")
        expected_keys = {
            "messages", "user_input", "observations", "sources",
            "citations", "artifacts", "reflection_items", "errors",
            "metadata", "governance", "sandbox", "orchestration",
        }
        assert set(state.keys()) == expected_keys
