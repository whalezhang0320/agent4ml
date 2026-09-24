from __future__ import annotations

import pytest

from agent4ml.backend.agents.memory.exceptions import (
    MemoryConflictError,
    MemoryConsolidateError,
    MemoryError,
    MemoryNotFoundError,
    MemoryRetrieveError,
    MemoryStoreError,
)


class TestMemoryErrorStr:
    def test_no_details_returns_message(self) -> None:
        err = MemoryError("something failed")
        assert str(err) == "something failed"

    def test_empty_details_returns_message(self) -> None:
        err = MemoryError("failed", details={})
        assert str(err) == "failed"

    def test_with_details_formats_key_val(self) -> None:
        err = MemoryError("failed", details={"key": "val"})
        assert str(err) == "failed (key='val')"

    def test_with_multiple_details(self) -> None:
        err = MemoryError("failed", details={"a": 1, "b": "x"})
        result = str(err)
        assert "failed" in result
        assert "a=1" in result
        assert "b='x'" in result

    def test_message_attribute(self) -> None:
        err = MemoryError("msg", details={"k": "v"})
        assert err.message == "msg"

    def test_details_attribute(self) -> None:
        err = MemoryError("msg", details={"k": "v"})
        assert err.details == {"k": "v"}

    def test_details_default_empty_dict(self) -> None:
        err = MemoryError("msg")
        assert err.details == {}


class TestMemoryNotFoundError:
    def test_includes_trace_id(self) -> None:
        err = MemoryNotFoundError("trace-abc")
        assert "trace-abc" in str(err)
        assert err.details["trace_id"] == "trace-abc"

    def test_is_memory_error(self) -> None:
        err = MemoryNotFoundError("x")
        assert isinstance(err, MemoryError)


class TestMemoryStoreError:
    def test_includes_operation_and_path(self) -> None:
        err = MemoryStoreError("write failed", operation="add", path="/tmp/mem.md")
        assert err.details["operation"] == "add"
        assert err.details["path"] == "/tmp/mem.md"

    def test_default_path_empty(self) -> None:
        err = MemoryStoreError("fail", operation="remove")
        assert err.details["path"] == ""

    def test_is_memory_error(self) -> None:
        err = MemoryStoreError("fail", operation="add")
        assert isinstance(err, MemoryError)


class TestMemoryRetrieveError:
    def test_short_query_not_truncated(self) -> None:
        err = MemoryRetrieveError("no result", query="short query")
        assert err.details["query"] == "short query"

    def test_long_query_truncated_to_103_chars(self) -> None:
        long_query = "x" * 150
        err = MemoryRetrieveError("fail", query=long_query)
        assert err.details["query"] == "x" * 100 + "..."
        assert len(err.details["query"]) == 103

    def test_boundary_100_chars_not_truncated(self) -> None:
        query = "x" * 100
        err = MemoryRetrieveError("fail", query=query)
        assert err.details["query"] == query

    def test_is_memory_error(self) -> None:
        err = MemoryRetrieveError("fail", query="q")
        assert isinstance(err, MemoryError)


class TestMemoryConsolidateError:
    def test_includes_trace_ids(self) -> None:
        err = MemoryConsolidateError("merge fail", trace_ids=["t1", "t2"])
        assert err.details["trace_ids"] == ["t1", "t2"]

    def test_is_memory_error(self) -> None:
        err = MemoryConsolidateError("fail", trace_ids=[])
        assert isinstance(err, MemoryError)


class TestMemoryConflictError:
    def test_includes_old_and_new_id(self) -> None:
        err = MemoryConflictError("conflict", old_id="old", new_id="new")
        assert err.details["old_id"] == "old"
        assert err.details["new_id"] == "new"

    def test_is_memory_error(self) -> None:
        err = MemoryConflictError("conflict", old_id="a", new_id="b")
        assert isinstance(err, MemoryError)
