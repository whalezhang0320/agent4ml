"""Stage 1 integration: Sandbox class composes runtime + translator + guard."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from agent4ml.backend.agents.capabilities.registry import (
    CapabilityMissingError,
    CapabilityRegistry,
)
from agent4ml.backend.agents.sandbox.contracts import (
    PathTranslator,
    SandboxProvider,
    SandboxRuntime,
    SecurityGuard,
)
from agent4ml.backend.agents.sandbox.sandbox import Sandbox
from agent4ml.backend.agents.sandbox.types import GrepMatch
from agent4ml.backend.agents.state.reducers import merge_sandbox


def _make_sandbox_with_mocks() -> tuple[Sandbox, MagicMock, MagicMock, MagicMock, list[str]]:
    """Build a Sandbox instance wired to MagicMock runtime/translator/guard, return calls list for assertion."""
    calls: list[str] = []
    runtime = MagicMock(spec=SandboxRuntime)
    translator = MagicMock(spec=PathTranslator)
    guard = MagicMock(spec=SecurityGuard)

    translator.translate_path.side_effect = lambda p: (calls.append("t.path"), p)[1]
    translator.translate_command.side_effect = lambda c: (calls.append("t.cmd"), c)[1]
    translator.mask_output.side_effect = lambda o: (calls.append("t.mask"), o)[1]
    guard.validate_path.side_effect = lambda p, *, write=False: calls.append(f"g.path(w={write})")
    guard.validate_command.side_effect = lambda cmd: calls.append("g.cmd")
    runtime.exec_command.side_effect = lambda c: (calls.append("r.exec"), "out")[1]

    sandbox = Sandbox("sb-int", runtime, translator, guard)
    return sandbox, runtime, translator, guard, calls


class TestSandboxOrchestrationIntegration:
    def test_execute_command_full_orchestration(self) -> None:
        sb, _, _, _, calls = _make_sandbox_with_mocks()
        result = sb.execute_command("ls /mnt/agent4ml/user-data")
        assert result == "out"
        assert calls == ["g.cmd", "t.cmd", "r.exec", "t.mask"]

    def test_guard_blocks_runtime_on_validation_failure(self) -> None:
        sb, runtime, _, guard, _ = _make_sandbox_with_mocks()
        guard.validate_command.side_effect = PermissionError("denied")
        with pytest.raises(PermissionError):
            sb.execute_command("ls")
        runtime.exec_command.assert_not_called()


class TestCapabilityRegistryIntegration:
    def test_registry_holds_sandbox_provider(self) -> None:
        mock_provider = MagicMock(spec=SandboxProvider)
        registry = CapabilityRegistry(sandbox_provider=mock_provider)
        assert registry.get_sandbox_provider() is mock_provider

    def test_registry_without_provider_raises(self) -> None:
        registry = CapabilityRegistry()
        with pytest.raises(CapabilityMissingError):
            registry.get_sandbox_provider()

    def test_registry_sandbox_provider_isolresearcher_model(self) -> None:
        mock_provider = MagicMock(spec=SandboxProvider)
        mock_model = object()
        registry = CapabilityRegistry(
            models={"researcher": mock_model},
            sandbox_provider=mock_provider,
        )
        assert registry.get_model("researcher") is mock_model
        assert registry.get_sandbox_provider() is mock_provider


class TestMergeSandboxGraphContext:
    """Verify LangGraph channel persistence emits sandbox_id."""

    def test_multiple_tools_same_id_idempotent(self) -> None:
        """Repeated merges with same sandbox_id collapse to single entry."""
        existing = None
        for _ in range(3):
            existing = merge_sandbox(existing, {"sandbox_id": "abc123"})
        assert existing == {"sandbox_id": "abc123"}

    def test_conflict_id_fail_closed(self) -> None:
        """Different id merge is bug, must fail-closed."""
        existing = {"sandbox_id": "abc"}
        with pytest.raises(ValueError, match="Conflicting"):
            merge_sandbox(existing, {"sandbox_id": "xyz"})

    def test_persistent_across_turns(self) -> None:
        """sandbox_id persists when a turn contributes no new sandbox patch."""
        turn1 = merge_sandbox(None, {"sandbox_id": "abc"})
        turn2 = merge_sandbox(turn1, None)
        assert turn2 == {"sandbox_id": "abc"}
        assert turn2 is turn1


class TestThreadStateBackwardCompat:
    def test_threadstate_extends_with_sandbox(self) -> None:
        """ThreadState sandbox field included in schema."""
        from agent4ml.backend.agents.state.types import ThreadState

        annotations = ThreadState.__annotations__
        assert "sandbox" in annotations
        assert "messages" in annotations
        assert "observations" in annotations
        assert "governance" in annotations
        assert "sources" in annotations
        assert "errors" in annotations
