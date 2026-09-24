from __future__ import annotations

from langchain_core.tools import BaseTool, tool

from agent4ml.backend.agents.sandbox.contracts import SandboxProvider
from agent4ml.backend.agents.sandbox.exceptions import (
    SandboxNotFoundError,
    SandboxRuntimeError,
)
from agent4ml.backend.agents.sandbox.integration.context import get_sandbox_id
from agent4ml.backend.agents.sandbox.sandbox import Sandbox
from agent4ml.backend.agents.sandbox.utils.file_operation_lock import (
    get_file_operation_lock,
)

_BASH_OUTPUT_MAX_CHARS = 10000
WRITE_FILE_MAX_BYTES = 5 * 1024 * 1024


def _truncate_output(
    output: str, max_chars: int = _BASH_OUTPUT_MAX_CHARS
) -> str:
    """截断长输出，超限加提示。"""
    if len(output) <= max_chars:
        return output
    return (
        output[:max_chars]
        + f"\n... (truncated, {len(output) - max_chars} chars omitted)"
    )


def _ensure_sandbox(provider: SandboxProvider) -> Sandbox:
    """从 ContextVar 取 sandbox_id + provider.get 获取 Sandbox。"""
    sandbox_id = get_sandbox_id()
    if sandbox_id is None:
        raise SandboxRuntimeError(
            "no sandbox in context (Stage 4 middleware not set)"
        )
    sandbox = provider.get(sandbox_id)
    if sandbox is None:
        raise SandboxNotFoundError(sandbox_id)
    return sandbox


def _make_bash_tool(provider: SandboxProvider) -> BaseTool:
    @tool("bash", parse_docstring=True)
    def bash_tool(command: str) -> str:
        """Execute a bash command in the sandbox.

        Args:
            command: The bash command to execute.
        """
        output = _ensure_sandbox(provider).execute_command(command)
        return _truncate_output(output)

    return bash_tool


def _make_read_file_tool(provider: SandboxProvider) -> BaseTool:
    @tool("read_file", parse_docstring=True)
    def read_file_tool(path: str) -> str:
        """Read the contents of a file.

        Args:
            path: Virtual path to the file (e.g. /mnt/agent4ml/user-data/workspace/file.txt).
        """
        return _ensure_sandbox(provider).read_file(path)

    return read_file_tool


def _make_write_file_tool(provider: SandboxProvider) -> BaseTool:
    @tool("write_file", parse_docstring=True)
    def write_file_tool(
        path: str, content: str, append: bool = False
    ) -> str:
        """Write content to a file. Creates parent directories if needed.

        Args:
            path: Virtual path to the file.
            content: Text content to write.
            append: If True, append to file; if False, overwrite.
        """
        if not append and len(content.encode("utf-8")) > WRITE_FILE_MAX_BYTES:
            raise SandboxRuntimeError(
                f"write_file content exceeds {WRITE_FILE_MAX_BYTES} bytes limit; "
                "use append=True to write in chunks"
            )
        _ensure_sandbox(provider).write_file(path, content, append=append)
        return f"wrote {len(content)} chars to {path}"

    return write_file_tool


def _make_list_dir_tool(provider: SandboxProvider) -> BaseTool:
    @tool("list_dir", parse_docstring=True)
    def list_dir_tool(path: str, max_depth: int = 2) -> str:
        """List directory contents in tree format.

        Args:
            path: Virtual path to the directory.
            max_depth: Maximum depth to traverse (default 2).
        """
        entries = _ensure_sandbox(provider).list_dir(path, max_depth=max_depth)
        if not entries:
            return "(empty)"
        lines: list[str] = []
        for entry in entries:
            parts = entry.replace("\\", "/").split("/")
            indent = "  " * (len(parts) - 1)
            lines.append(f"{indent}{parts[-1]}")
        return "\n".join(lines)

    return list_dir_tool


def _make_str_replace_tool(provider: SandboxProvider) -> BaseTool:
    @tool("str_replace", parse_docstring=True)
    def str_replace_tool(
        path: str, old_str: str, new_str: str, replace_all: bool = False
    ) -> str:
        """Replace text in a file. Serialized per (sandbox, path).

        Args:
            path: Virtual path to the file.
            old_str: Text to find.
            new_str: Replacement text.
            replace_all: If True, replace all occurrences; if False, replace first.
        """
        sandbox = _ensure_sandbox(provider)
        lock = get_file_operation_lock(sandbox.id, path)
        with lock:
            content = sandbox.read_file(path)
            if old_str not in content:
                raise SandboxRuntimeError(f"old_str not found in {path}")
            count = content.count(old_str)
            if replace_all:
                new_content = content.replace(old_str, new_str)
            else:
                new_content = content.replace(old_str, new_str, 1)
            sandbox.write_file(path, new_content)
        return f"replaced {count if replace_all else 1} occurrence(s) in {path}"

    return str_replace_tool


def make_sandbox_tools(provider: SandboxProvider, allow_host_bash: bool = True) -> list[BaseTool]:
    """工厂：构造 sandbox 工具集，闭包捕获 provider。

    返回 6 个 @tool：bash / read_file / write_file / list_dir / str_replace / present_files。
    工具内部用 ContextVar get_sandbox_id 获取 sandbox_id + provider.get 获取 Sandbox。
    allow_host_bash=False 时不注册 bash 工具（S2 安全加固）。
    """
    tools = []
    if allow_host_bash:
        tools.append(_make_bash_tool(provider))
    tools.extend([
        _make_read_file_tool(provider),
        _make_write_file_tool(provider),
        _make_list_dir_tool(provider),
        _make_str_replace_tool(provider),
        _make_present_files_tool(provider),
    ])
    return tools


def _make_present_files_tool(provider: SandboxProvider) -> BaseTool:
    @tool("present_files", parse_docstring=True)
    def present_files_tool(paths: list[str]) -> str:
        """Declare files as deliverable artifacts. The system generates downloadable URLs for the user to access these files via browser.

        Call this after generating final output files (pptx, csv, py, pdf, etc.) in the sandbox.
        Files MUST be under /mnt/agent4ml/user-data/ (e.g. /mnt/agent4ml/user-data/workspace/report.pptx).
        Files in other paths (e.g. /home/...) are not accessible to the host and cannot be delivered.

        Args:
            paths: Virtual paths to files under /mnt/agent4ml/user-data/ (e.g. /mnt/agent4ml/user-data/workspace/report.pptx).
        """
        _REQUIRED_PREFIX = "/mnt/agent4ml/user-data/"
        invalid = [p for p in paths if not p.startswith(_REQUIRED_PREFIX)]
        if invalid:
            raise SandboxRuntimeError(
                f"present_files: paths must be under {_REQUIRED_PREFIX}. "
                f"Invalid: {invalid}. Copy files there first with: "
                f"bash('cp <src> /mnt/agent4ml/user-data/workspace/<filename>')"
            )
        return f"Presented {len(paths)} file(s): {', '.join(paths)}"

    return present_files_tool
