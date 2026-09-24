from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent4ml.backend.agents.sandbox.contracts import (
    PathTranslator,
    SandboxRuntime,
    SandboxProvider,
    SecurityGuard,
)
from agent4ml.backend.agents.sandbox.exceptions import (
    SandboxNotFoundError,
    SandboxRuntimeError,
)
from agent4ml.backend.agents.sandbox.integration.context import set_sandbox_id
from agent4ml.backend.agents.sandbox.integration.tools import (
    _ensure_sandbox,
    _truncate_output,
    _make_bash_tool,
    _make_read_file_tool,
    _make_write_file_tool,
    _make_list_dir_tool,
    _make_str_replace_tool,
    make_sandbox_tools,
    WRITE_FILE_MAX_BYTES,
)
from agent4ml.backend.agents.sandbox.sandbox import Sandbox


def _make_mock_sandbox() -> MagicMock:
    """Build a MagicMock conforming to Sandbox spec with fixed id 'sb-mock'."""
    sandbox = MagicMock(spec=Sandbox)
    sandbox.id = "sb-mock"
    return sandbox


def _make_mock_provider(sandbox: Sandbox | None = None) -> MagicMock:
    """Build a MagicMock SandboxProvider whose get() returns the supplied sandbox."""
    provider = MagicMock(spec=SandboxProvider)
    provider.get.return_value = sandbox
    return provider


class TestTruncateOutput:
    def test_short_output_passthrough(self) -> None:
        assert _truncate_output("short") == "short"

    def test_exact_limit_passthrough(self) -> None:
        output = "x" * 10000
        assert _truncate_output(output) == output

    def test_over_limit_truncated(self) -> None:
        output = "x" * 15000
        result = _truncate_output(output)
        assert result[:10000] == "x" * 10000
        assert "truncated" in result
        assert "5000 chars omitted" in result


class TestEnsureSandbox:
    def test_no_sandbox_id_raises(self) -> None:
        set_sandbox_id(None)
        provider = _make_mock_provider()
        with pytest.raises(SandboxRuntimeError, match="no sandbox in context"):
            _ensure_sandbox(provider)

    def test_sandbox_not_found_raises(self) -> None:
        set_sandbox_id("sb-ghost")
        provider = _make_mock_provider(sandbox=None)
        with pytest.raises(SandboxNotFoundError):
            _ensure_sandbox(provider)

    def test_returns_sandbox(self) -> None:
        set_sandbox_id("sb-1")
        mock_sb = _make_mock_sandbox()
        provider = _make_mock_provider(sandbox=mock_sb)
        result = _ensure_sandbox(provider)
        assert result is mock_sb
        provider.get.assert_called_once_with("sb-1")


class TestBashTool:
    def test_execute_command(self) -> None:
        set_sandbox_id("sb-1")
        mock_sb = _make_mock_sandbox()
        mock_sb.execute_command.return_value = "hello\n"
        provider = _make_mock_provider(sandbox=mock_sb)

        bash = _make_bash_tool(provider)
        result = bash.invoke({"command": "echo hello"})
        assert result == "hello\n"
        mock_sb.execute_command.assert_called_once_with("echo hello")

    def test_long_output_truncated(self) -> None:
        set_sandbox_id("sb-1")
        mock_sb = _make_mock_sandbox()
        mock_sb.execute_command.return_value = "x" * 15000
        provider = _make_mock_provider(sandbox=mock_sb)

        bash = _make_bash_tool(provider)
        result = bash.invoke({"command": "cat bigfile"})
        assert len(result) < 15000
        assert "truncated" in result

    def test_no_sandbox_id_raises(self) -> None:
        set_sandbox_id(None)
        provider = _make_mock_provider()
        bash = _make_bash_tool(provider)
        with pytest.raises(SandboxRuntimeError):
            bash.invoke({"command": "ls"})


class TestReadFileTool:
    def test_read_file(self) -> None:
        set_sandbox_id("sb-1")
        mock_sb = _make_mock_sandbox()
        mock_sb.read_file.return_value = "file content"
        provider = _make_mock_provider(sandbox=mock_sb)

        read_file = _make_read_file_tool(provider)
        result = read_file.invoke({"path": "/mnt/agent4ml/user-data/workspace/file.txt"})
        assert result == "file content"
        mock_sb.read_file.assert_called_once_with("/mnt/agent4ml/user-data/workspace/file.txt")

    def test_no_sandbox_id_raises(self) -> None:
        set_sandbox_id(None)
        provider = _make_mock_provider()
        read_file = _make_read_file_tool(provider)
        with pytest.raises(SandboxRuntimeError):
            read_file.invoke({"path": "/mnt/x"})


class TestWriteFileTool:
    def test_write(self) -> None:
        set_sandbox_id("sb-1")
        mock_sb = _make_mock_sandbox()
        provider = _make_mock_provider(sandbox=mock_sb)

        write_file = _make_write_file_tool(provider)
        result = write_file.invoke({"path": "/mnt/x", "content": "data"})
        mock_sb.write_file.assert_called_once_with("/mnt/x", "data", append=False)
        assert "wrote" in result

    def test_append(self) -> None:
        set_sandbox_id("sb-1")
        mock_sb = _make_mock_sandbox()
        provider = _make_mock_provider(sandbox=mock_sb)

        write_file = _make_write_file_tool(provider)
        write_file.invoke({"path": "/mnt/x", "content": "data", "append": True})
        mock_sb.write_file.assert_called_once_with("/mnt/x", "data", append=True)

    def test_over_80kb_non_append_rejected(self) -> None:
        set_sandbox_id("sb-1")
        mock_sb = _make_mock_sandbox()
        provider = _make_mock_provider(sandbox=mock_sb)

        write_file = _make_write_file_tool(provider)
        big_content = "x" * (WRITE_FILE_MAX_BYTES + 1)
        with pytest.raises(SandboxRuntimeError, match="exceeds"):
            write_file.invoke({"path": "/mnt/x", "content": big_content})

    def test_over_80kb_append_allowed(self) -> None:
        set_sandbox_id("sb-1")
        mock_sb = _make_mock_sandbox()
        provider = _make_mock_provider(sandbox=mock_sb)

        write_file = _make_write_file_tool(provider)
        big_content = "x" * (WRITE_FILE_MAX_BYTES + 1)
        write_file.invoke({"path": "/mnt/x", "content": big_content, "append": True})
        mock_sb.write_file.assert_called_once()


class TestListDirTool:
    def test_empty_dir(self) -> None:
        set_sandbox_id("sb-1")
        mock_sb = _make_mock_sandbox()
        mock_sb.list_dir.return_value = []
        provider = _make_mock_provider(sandbox=mock_sb)

        list_dir = _make_list_dir_tool(provider)
        result = list_dir.invoke({"path": "/mnt/x"})
        assert result == "(empty)"

    def test_tree_format(self) -> None:
        set_sandbox_id("sb-1")
        mock_sb = _make_mock_sandbox()
        mock_sb.list_dir.return_value = ["a.txt", "sub/b.txt"]
        provider = _make_mock_provider(sandbox=mock_sb)

        list_dir = _make_list_dir_tool(provider)
        result = list_dir.invoke({"path": "/mnt/x"})
        lines = result.split("\n")
        assert "a.txt" in lines
        assert any("b.txt" in line and line.startswith("  ") for line in lines)

    def test_max_depth_passed(self) -> None:
        set_sandbox_id("sb-1")
        mock_sb = _make_mock_sandbox()
        mock_sb.list_dir.return_value = []
        provider = _make_mock_provider(sandbox=mock_sb)

        list_dir = _make_list_dir_tool(provider)
        list_dir.invoke({"path": "/mnt/x", "max_depth": 3})
        mock_sb.list_dir.assert_called_once_with("/mnt/x", max_depth=3)


class TestStrReplaceTool:
    def test_single_replace(self) -> None:
        set_sandbox_id("sb-1")
        mock_sb = _make_mock_sandbox()
        mock_sb.read_file.return_value = "hello world hello"
        provider = _make_mock_provider(sandbox=mock_sb)

        str_replace = _make_str_replace_tool(provider)
        result = str_replace.invoke({
            "path": "/mnt/x", "old_str": "hello", "new_str": "hi"
        })
        mock_sb.read_file.assert_called_once_with("/mnt/x")
        mock_sb.write_file.assert_called_once_with("/mnt/x", "hi world hello")
        assert "1 occurrence" in result

    def test_replace_all(self) -> None:
        set_sandbox_id("sb-1")
        mock_sb = _make_mock_sandbox()
        mock_sb.read_file.return_value = "hello world hello"
        provider = _make_mock_provider(sandbox=mock_sb)

        str_replace = _make_str_replace_tool(provider)
        result = str_replace.invoke({
            "path": "/mnt/x", "old_str": "hello", "new_str": "hi", "replace_all": True
        })
        mock_sb.write_file.assert_called_once_with("/mnt/x", "hi world hi")
        assert "2 occurrence" in result

    def test_old_str_not_found(self) -> None:
        set_sandbox_id("sb-1")
        mock_sb = _make_mock_sandbox()
        mock_sb.read_file.return_value = "no match here"
        provider = _make_mock_provider(sandbox=mock_sb)

        str_replace = _make_str_replace_tool(provider)
        with pytest.raises(SandboxRuntimeError, match="not found"):
            str_replace.invoke({
                "path": "/mnt/x", "old_str": "missing", "new_str": "x"
            })


class TestMakeSandboxTools:
    def test_returns_6_tools(self) -> None:
        provider = _make_mock_provider()
        tools = make_sandbox_tools(provider)
        assert len(tools) == 6

    def test_tool_names(self) -> None:
        provider = _make_mock_provider()
        tools = make_sandbox_tools(provider)
        names = {t.name for t in tools}
        assert names == {"bash", "read_file", "write_file", "list_dir", "str_replace", "present_files"}

    def test_all_are_base_tool(self) -> None:
        from langchain_core.tools import BaseTool
        provider = _make_mock_provider()
        tools = make_sandbox_tools(provider)
        assert all(isinstance(t, BaseTool) for t in tools)
