"""Tests for interrupt protection utility."""

from __future__ import annotations

from agent4ml.backend.agents.observability.interrupt_protection import (
    interrupt_protection,
    is_interrupt_protected,
)


class TestInterruptProtection:
    def test_default_unprotected(self) -> None:
        assert not is_interrupt_protected()

    def test_protected_inside_context(self) -> None:
        with interrupt_protection():
            assert is_interrupt_protected()

    def test_unprotected_after_context(self) -> None:
        with interrupt_protection():
            pass
        assert not is_interrupt_protected()

    def test_nested_protection(self) -> None:
        with interrupt_protection():
            assert is_interrupt_protected()
            with interrupt_protection(False):
                assert not is_interrupt_protected()
            assert is_interrupt_protected()

    def test_exception_still_restores_state(self) -> None:
        try:
            with interrupt_protection():
                raise RuntimeError("test")
        except RuntimeError:
            pass
        assert not is_interrupt_protected()
