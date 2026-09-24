from __future__ import annotations

from agent4ml.backend.agents.sandbox.contracts import SecurityGuard


class _CompleteGuard:
    """Mock 实现 SecurityGuard 全部 2 方法。无 mask_output。"""

    def validate_path(self, path: str, *, write: bool = False) -> None:
        pass

    def validate_command(self, command: str) -> None:
        pass


class _GuardWithMaskOutput:
    """多了 mask_output 方法（Guard 不该有，但 Protocol 不禁止多余方法）。"""

    def validate_path(self, path: str, *, write: bool = False) -> None:
        pass

    def validate_command(self, command: str) -> None:
        pass

    def mask_output(self, output: str) -> str:
        return output


class _IncompleteGuard:
    """缺 validate_command 方法。"""
    def validate_path(self, path: str, *, write: bool = False) -> None:
        pass


class TestSecurityGuardProtocol:
    def test_complete_impl_is_instance(self) -> None:
        guard = _CompleteGuard()
        assert isinstance(guard, SecurityGuard)

    def test_incomplete_impl_not_instance(self) -> None:
        guard = _IncompleteGuard()
        assert not isinstance(guard, SecurityGuard)

    def test_guard_no_mask_output_required(self) -> None:
        """SecurityGuard Protocol 不要求 mask_output（Grill #4）。"""
        guard = _CompleteGuard()
        assert not hasattr(guard, "mask_output") or callable(getattr(guard, "mask_output", None)) is False
        assert isinstance(guard, SecurityGuard)

    def test_extra_methods_allowed(self) -> None:
        """Protocol 允许实现类有额外方法（结构化类型）。"""
        guard = _GuardWithMaskOutput()
        assert isinstance(guard, SecurityGuard)
