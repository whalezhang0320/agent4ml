from __future__ import annotations

from agent4ml.backend.agents.sandbox.contracts import SecurityGuard
from agent4ml.backend.agents.sandbox.guards.permissive_guard import PermissiveGuard


class TestPermissiveGuard:
    def test_validate_path_no_raise(self) -> None:
        guard = PermissiveGuard()
        guard.validate_path("/any/path")

    def test_validate_path_write_no_raise(self) -> None:
        guard = PermissiveGuard()
        guard.validate_path("/any/path", write=True)

    def test_validate_command_no_raise(self) -> None:
        guard = PermissiveGuard()
        guard.validate_command("rm -rf /")

    def test_is_security_guard(self) -> None:
        guard = PermissiveGuard()
        assert isinstance(guard, SecurityGuard)

    def test_no_mask_output(self) -> None:
        """PermissiveGuard does not implement mask_output (no output rewriting)."""
        guard = PermissiveGuard()
        assert not hasattr(guard, "mask_output")

    def test_traversal_path_no_raise(self) -> None:
        """PermissiveGuard accepts path traversal sequences (no sanitization by design)."""
        guard = PermissiveGuard()
        guard.validate_path("/mnt/agent4ml/user-data/../../../etc/passwd")
