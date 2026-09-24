from __future__ import annotations

from agent4ml.backend.agents.sandbox.integration.context import (
    get_sandbox_id,
    set_sandbox_id,
)


class TestContextVar:
    def test_default_none(self) -> None:
        set_sandbox_id(None)
        assert get_sandbox_id() is None

    def test_set_then_get(self) -> None:
        set_sandbox_id("sb-1")
        assert get_sandbox_id() == "sb-1"

    def test_reset_to_none(self) -> None:
        set_sandbox_id("sb-1")
        set_sandbox_id(None)
        assert get_sandbox_id() is None

    def test_overwrite(self) -> None:
        set_sandbox_id("sb-1")
        set_sandbox_id("sb-2")
        assert get_sandbox_id() == "sb-2"
