from __future__ import annotations

from agent4ml.backend.agents.sandbox.contracts import SandboxRuntime
from agent4ml.backend.agents.sandbox.types import GrepMatch


class _CompleteRuntime:
    """Mock 实现 SandboxRuntime 全部 9 方法。"""

    def exec_command(self, command: str) -> str:
        return ""

    def read_file(self, path: str) -> str:
        return ""

    def write_file(self, path: str, content: str, append: bool = False) -> None:
        pass

    def list_dir(self, path: str, max_depth: int = 2) -> list[str]:
        return []

    def glob(
        self, path: str, pattern: str, *, include_dirs: bool = False, max_results: int = 200
    ) -> tuple[list[str], bool]:
        return [], False

    def grep(
        self, path: str, pattern: str, *, glob: str | None = None,
        literal: bool = False, case_sensitive: bool = False, max_results: int = 100,
    ) -> tuple[list[GrepMatch], bool]:
        return [], False

    def download_file(self, path: str) -> bytes:
        return b""

    def update_file(self, path: str, content: bytes) -> None:
        pass

    def close(self) -> None:
        pass


class _IncompleteRuntime:
    """缺 exec_command 方法。"""
    def read_file(self, path: str) -> str:
        return ""


class TestSandboxRuntimeProtocol:
    def test_complete_impl_is_instance(self) -> None:
        runtime = _CompleteRuntime()
        assert isinstance(runtime, SandboxRuntime)

    def test_incomplete_impl_not_instance(self) -> None:
        runtime = _IncompleteRuntime()
        assert not isinstance(runtime, SandboxRuntime)

    def test_structural_typing_no_inheritance(self) -> None:
        runtime = _CompleteRuntime()
        assert not isinstance(runtime, type) or SandboxRuntime not in type(runtime).__mro__
