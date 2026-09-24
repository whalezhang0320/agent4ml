from __future__ import annotations

from dataclasses import replace

from agent4ml.backend.agents.sandbox.contracts import (
    PathTranslator,
    SandboxRuntime,
    SecurityGuard,
)
from agent4ml.backend.agents.sandbox.types import GrepMatch


class Sandbox:
    """Sandbox 具体类（非 ABC），组合 Runtime + Translator + Guard，负责编排。

    方案 C 核心。编排流程：validate → translate → execute → mask。
    切沙箱（Local/Docker/E2B）只换组件，编排逻辑复用。

    INVARIANT:
    - 所有操作方法遵守编排顺序：guard.validate → translator.translate → runtime.execute → translator.mask
    - mask_output 归 translator（路径翻译逆操作），guard 只做 validate（Grill #4）
    - close() 后不得再调用操作方法
    - id 在构造时确定，不可变
    """

    def __init__(
        self,
        id: str,
        runtime: SandboxRuntime,
        translator: PathTranslator,
        guard: SecurityGuard,
    ) -> None:
        self._id = id
        self._runtime = runtime
        self._translator = translator
        self._guard = guard

    @property
    def id(self) -> str:
        return self._id

    def get_host_path(self, virtual_path: str) -> str:
        """虚拟路径 → 宿主物理路径（供外部复制/注册 artifact 用）。

        优先调 translator.reverse_translate（DockerPathTranslator 有）；
        fallback translate_path（IdentityTranslator / LocalPathTranslator 无 reverse）。
        不做 guard.validate（调用方负责路径安全）。
        """
        if hasattr(self._translator, "reverse_translate"):
            return self._translator.reverse_translate(virtual_path)
        return self._translator.translate_path(virtual_path)

    def execute_command(self, command: str) -> str:
        self._guard.validate_command(command)
        translated = self._translator.translate_command(command)
        output = self._runtime.exec_command(translated)
        return self._translator.mask_output(output)

    def read_file(self, path: str) -> str:
        self._guard.validate_path(path, write=False)
        physical = self._translator.translate_path(path)
        content = self._runtime.read_file(physical)
        return self._translator.mask_output(content)

    def write_file(self, path: str, content: str, append: bool = False) -> None:
        self._guard.validate_path(path, write=True)
        physical = self._translator.translate_path(path)
        self._runtime.write_file(physical, content, append=append)

    def list_dir(self, path: str, max_depth: int = 2, max_entries: int = 1000) -> list[str]:
        self._guard.validate_path(path, write=False)
        physical = self._translator.translate_path(path)
        entries = self._runtime.list_dir(physical, max_depth=max_depth, max_entries=max_entries)
        return [self._translator.mask_output(e) for e in entries]

    def glob(
        self,
        path: str,
        pattern: str,
        *,
        include_dirs: bool = False,
        max_results: int = 200,
    ) -> tuple[list[str], bool]:
        self._guard.validate_path(path, write=False)
        physical = self._translator.translate_path(path)
        results, truncated = self._runtime.glob(
            physical, pattern, include_dirs=include_dirs, max_results=max_results
        )
        return [self._translator.mask_output(r) for r in results], truncated

    def grep(
        self,
        path: str,
        pattern: str,
        *,
        glob: str | None = None,
        literal: bool = False,
        case_sensitive: bool = False,
        max_results: int = 100,
    ) -> tuple[list[GrepMatch], bool]:
        self._guard.validate_path(path, write=False)
        physical = self._translator.translate_path(path)
        results, truncated = self._runtime.grep(
            physical,
            pattern,
            glob=glob,
            literal=literal,
            case_sensitive=case_sensitive,
            max_results=max_results,
        )
        masked_results = [
            replace(
                r,
                path=self._translator.mask_output(r.path),
                line=self._translator.mask_output(r.line),
            )
            for r in results
        ]
        return masked_results, truncated

    def download_file(self, path: str) -> bytes:
        self._guard.validate_path(path, write=False)
        physical = self._translator.translate_path(path)
        return self._runtime.download_file(physical)

    def update_file(self, path: str, content: bytes) -> None:
        self._guard.validate_path(path, write=True)
        physical = self._translator.translate_path(path)
        self._runtime.update_file(physical, content)

    def close(self) -> None:
        self._runtime.close()
