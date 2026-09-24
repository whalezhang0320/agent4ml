from __future__ import annotations

import fnmatch
import os
import re
import subprocess
from pathlib import Path

from agent4ml.backend.agents.sandbox.exceptions import (
    SandboxCommandError,
    SandboxFileNotFoundError,
    SandboxPermissionError,
    SandboxRuntimeError,
)
from agent4ml.backend.agents.sandbox.types import GrepMatch
from agent4ml.backend.agents.sandbox.utils.search import (
    DEFAULT_LINE_SUMMARY_LENGTH,
    DEFAULT_MAX_FILE_SIZE_BYTES,
    IGNORE_PATTERNS,
)

_EXEC_TIMEOUT_SECONDS = 600

# S10: ReDoS 防护
_MAX_REGEX_PATTERN_LENGTH = 200
# 检测嵌套量词：(group with quantifier) followed by quantifier — ReDoS 经典模式
_NESTED_QUANTIFIER_RE = re.compile(r"\([^)]*[+*?][^)]*\)[+*?]")


def _validate_regex_pattern(pattern: str) -> None:
    """ReDoS 防护：拒绝超长 pattern + 嵌套量词（如 (a+)+）。

    Python re 无 timeout 机制，用户控制 pattern + 无防护 = 灾难回溯 DoS。
    """
    if len(pattern) > _MAX_REGEX_PATTERN_LENGTH:
        raise ValueError(
            f"regex pattern too long ({len(pattern)} > {_MAX_REGEX_PATTERN_LENGTH} chars)"
        )
    if _NESTED_QUANTIFIER_RE.search(pattern):
        raise ValueError(
            f"potential ReDoS pattern (nested quantifier): {pattern[:50]}"
        )


def _is_path_ignored(file_path: Path, root: Path) -> bool:
    """检查路径的任一部分是否匹配 IGNORE_PATTERNS（含目录名如 .git）。"""
    try:
        rel = file_path.relative_to(root)
    except ValueError:
        return False
    for part in rel.parts:
        for pat in IGNORE_PATTERNS:
            if fnmatch.fnmatch(part, pat):
                return True
    return False


class LocalRuntime:
    """LocalRuntime — subprocess 裸执行 + Python 标准库文件操作。

    方案 C 三组件之一。只负责裸执行，不知路径翻译、不做安全检查。
    全抛 SandboxError 子类（Grill #5），包装内置异常。

    INVARIANT:
    - exec_command 用 subprocess.run(shell=True)，捕获 CalledProcessError → SandboxCommandError
    - allow_host_bash=False 时 exec_command 直接 raise SandboxPermissionError（S2 安全加固）
    - 文件操作捕获 FileNotFoundError → SandboxFileNotFoundError，PermissionError → SandboxPermissionError
    - grep 用 IGNORE_PATTERNS 过滤 + DEFAULT_MAX_FILE_SIZE_BYTES 跳过大文件
    - close no-op（subprocess 无持久连接）
    - 无 write_file 80KB 限制（Grill #7：裸执行不关心 LLM chunk 大小，Stage 3 工具层加）
    """

    def __init__(self, allow_host_bash: bool = True) -> None:
        self._allow_host_bash = allow_host_bash

    def exec_command(self, command: str) -> str:
        if not self._allow_host_bash:
            raise SandboxRuntimeError(
                "host bash is disabled (AGENT4ML_SANDBOX_ALLOW_HOST_BASH=false)"
            )
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=_EXEC_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise SandboxCommandError(
                "command timed out", command=command, exit_code=None
            ) from exc
        if result.returncode != 0:
            raise SandboxCommandError(
                f"command failed with exit code {result.returncode}",
                command=command,
                exit_code=result.returncode,
            )
        return result.stdout

    def read_file(self, path: str) -> str:
        try:
            return Path(path).read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise SandboxFileNotFoundError(
                f"file not found: {path}", path=path, operation="read"
            ) from exc
        except PermissionError as exc:
            raise SandboxPermissionError(
                f"permission denied: {path}", path=path, operation="read"
            ) from exc

    def write_file(self, path: str, content: str, append: bool = False) -> None:
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            if append:
                with p.open("a", encoding="utf-8") as f:
                    f.write(content)
            else:
                p.write_text(content, encoding="utf-8")
        except PermissionError as exc:
            raise SandboxPermissionError(
                f"permission denied: {path}", path=path, operation="write"
            ) from exc

    def list_dir(self, path: str, max_depth: int = 2, max_entries: int = 1000) -> list[str]:
        """BFS 剪枝遍历——os.scandir 按层下钻，超 max_depth 不递归，超 max_entries 截断。

        S9: 替代 rglob("*") 先全遍历再过滤的反模式——node_modules 10 万文件不再 DoS。
        """
        try:
            root = Path(path)
            if not root.exists():
                raise SandboxFileNotFoundError(
                    f"dir not found: {path}", path=path, operation="list_dir"
                )
            entries: list[str] = []
            self._scan_bfs(root, root, 1, max_depth, max_entries, entries)
            return sorted(entries)
        except PermissionError as exc:
            raise SandboxPermissionError(
                f"permission denied: {path}", path=path, operation="list_dir"
            ) from exc

    @staticmethod
    def _scan_bfs(
        root: Path, current: Path, depth: int, max_depth: int,
        max_entries: int, entries: list[str],
    ) -> None:
        """BFS 递归扫描——depth = 当前层条目的路径段数（root 直接子项 depth=1）。"""
        if depth > max_depth or len(entries) >= max_entries:
            return
        try:
            with os.scandir(current) as it:
                for entry in sorted(it, key=lambda e: e.name):
                    if len(entries) >= max_entries:
                        return
                    rel = str(Path(entry.path).relative_to(root))
                    entries.append(rel)
                    if entry.is_dir() and depth < max_depth:
                        LocalRuntime._scan_bfs(
                            root, Path(entry.path), depth + 1, max_depth, max_entries, entries,
                        )
        except (PermissionError, OSError):
            pass  # 跳过不可读目录

    def glob(
        self,
        path: str,
        pattern: str,
        *,
        include_dirs: bool = False,
        max_results: int = 200,
    ) -> tuple[list[str], bool]:
        try:
            root = Path(path)
            matches: list[str] = []
            for item in root.rglob(pattern):
                if not include_dirs and item.is_dir():
                    continue
                matches.append(str(item.relative_to(root)))
                if len(matches) >= max_results:
                    return matches, True
            return matches, False
        except PermissionError as exc:
            raise SandboxPermissionError(
                f"permission denied: {path}", path=path, operation="glob"
            ) from exc

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
        try:
            root = Path(path)
            flags = 0 if case_sensitive else re.IGNORECASE
            if literal:
                regex = re.compile(re.escape(pattern), flags)
            else:
                # S10: ReDoS 防护——非 literal 模式校验 pattern
                _validate_regex_pattern(pattern)
                regex = re.compile(pattern, flags)
            matches: list[GrepMatch] = []
            for file_path in root.rglob(glob or "*"):
                if file_path.is_dir():
                    continue
                if _is_path_ignored(file_path, root):
                    continue
                try:
                    stat = file_path.stat()
                except OSError:
                    continue
                if stat.st_size > DEFAULT_MAX_FILE_SIZE_BYTES:
                    continue
                try:
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                except (PermissionError, OSError):
                    continue
                for line_num, line in enumerate(content.splitlines(), start=1):
                    if regex.search(line):
                        truncated = line[:DEFAULT_LINE_SUMMARY_LENGTH]
                        matches.append(
                            GrepMatch(str(file_path), line_num, truncated)
                        )
                        if len(matches) >= max_results:
                            return matches, True
            return matches, False
        except PermissionError as exc:
            raise SandboxPermissionError(
                f"permission denied: {path}", path=path, operation="grep"
            ) from exc

    def download_file(self, path: str) -> bytes:
        try:
            return Path(path).read_bytes()
        except FileNotFoundError as exc:
            raise SandboxFileNotFoundError(
                f"file not found: {path}", path=path, operation="download"
            ) from exc

    def update_file(self, path: str, content: bytes) -> None:
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(content)
        except PermissionError as exc:
            raise SandboxPermissionError(
                f"permission denied: {path}", path=path, operation="update"
            ) from exc

    def close(self) -> None:
        pass
