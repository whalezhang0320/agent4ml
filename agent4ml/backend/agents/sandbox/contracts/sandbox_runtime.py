from __future__ import annotations

from typing import Protocol, runtime_checkable

from agent4ml.backend.agents.sandbox.types import GrepMatch


@runtime_checkable
class SandboxRuntime(Protocol):
    """中立沙箱运行时协议（裸执行契约）。

    方案 C 三组件之一。只负责裸执行（exec/read/write），不知路径翻译、不做安全检查。
    Local / Docker / E2B 各写 adapter。异构差异（E2B 无 glob）封装在 adapter 内。

    所有方法遵守调用方传入的路径契约（Local 路径已翻译为物理路径；Docker/E2B 直传虚拟路径）。
    异常类型：全抛 SandboxError 子类（SandboxCommandError / SandboxFileError /
    SandboxPermissionError / SandboxFileNotFoundError / SandboxRuntimeError）。
    runtime 实现负责包装内置异常（subprocess.CalledProcessError → SandboxCommandError，
    FileNotFoundError → SandboxFileNotFoundError 等）。
    工具层只需 catch SandboxError，由 ToolCallMiddleware 统一转 error ToolMessage。
    """

    def exec_command(self, command: str) -> str: ...

    def read_file(self, path: str) -> str: ...

    def write_file(self, path: str, content: str, append: bool = False) -> None: ...

    def list_dir(self, path: str, max_depth: int = 2, max_entries: int = 1000) -> list[str]: ...

    def glob(
        self,
        path: str,
        pattern: str,
        *,
        include_dirs: bool = False,
        max_results: int = 200,
    ) -> tuple[list[str], bool]: ...

    def grep(
        self,
        path: str,
        pattern: str,
        *,
        glob: str | None = None,
        literal: bool = False,
        case_sensitive: bool = False,
        max_results: int = 100,
    ) -> tuple[list[GrepMatch], bool]: ...

    def download_file(self, path: str) -> bytes: ...

    def update_file(self, path: str, content: bytes) -> None: ...

    def close(self) -> None: ...
