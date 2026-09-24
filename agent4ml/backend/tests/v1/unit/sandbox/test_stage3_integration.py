"""Stage 3 integration: LocalSandboxProvider + tools + group filtering."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent4ml.backend.agents.agent_tools.available import (
    SANDBOX_TOOL_NAMES,
    _tool_group,
)
from agent4ml.backend.agents.sandbox.integration.context import set_sandbox_id
from agent4ml.backend.agents.sandbox.integration.tools import make_sandbox_tools
from agent4ml.backend.agents.sandbox.local.local_sandbox_provider import (
    LocalSandboxProvider,
)
from agent4ml.backend.agents.sandbox.types import PathMapping


@pytest.fixture
def provider(tmp_path) -> LocalSandboxProvider:
    ws = tmp_path / "ws"
    ws.mkdir()
    mappings = [PathMapping("/mnt/agent4ml/user-data/workspace", str(ws))]
    return LocalSandboxProvider(mappings)


@pytest.fixture
def tools(provider: LocalSandboxProvider):
    return make_sandbox_tools(provider)


@pytest.fixture
def sandbox_id(provider: LocalSandboxProvider):
    sid = provider.acquire("thread-test", user_id="user-test")
    set_sandbox_id(sid)
    yield sid
    set_sandbox_id(None)


class TestGroupFiltering:
    def test_sandbox_group_classification(self) -> None:
        assert _tool_group("bash") == "sandbox"
        assert _tool_group("read_file") == "sandbox"
        assert _tool_group("write_file") == "sandbox"
        assert _tool_group("list_dir") == "sandbox"
        assert _tool_group("str_replace") == "sandbox"

    def test_core_group_unchanged(self) -> None:
        assert _tool_group("web_search") == "core"
        assert _tool_group("browse_page") == "core"
        assert _tool_group("read_snapshot") == "core"

    def test_deferred_group_unchanged(self) -> None:
        assert _tool_group("deep_search") == "deferred"
        assert _tool_group("get_page_links") == "deferred"

    def test_sandbox_tool_names_complete(self) -> None:
        assert SANDBOX_TOOL_NAMES == {
            "bash", "read_file", "write_file", "list_dir", "str_replace", "present_files"
        }


class TestEndToEndWithLocalProvider:
    def test_bash_echo(self, tools, sandbox_id) -> None:
        bash = next(t for t in tools if t.name == "bash")
        result = bash.invoke({"command": "echo hello"})
        assert "hello" in result

    def test_write_then_read(self, tools, sandbox_id) -> None:
        write_file = next(t for t in tools if t.name == "write_file")
        read_file = next(t for t in tools if t.name == "read_file")

        write_file.invoke({
            "path": "/mnt/agent4ml/user-data/workspace/test.txt",
            "content": "test content",
        })
        result = read_file.invoke({"path": "/mnt/agent4ml/user-data/workspace/test.txt"})
        assert result == "test content"

    def test_list_dir_shows_written_file(self, tools, sandbox_id) -> None:
        write_file = next(t for t in tools if t.name == "write_file")
        list_dir = next(t for t in tools if t.name == "list_dir")

        write_file.invoke({
            "path": "/mnt/agent4ml/user-data/workspace/nested/file.txt",
            "content": "x",
        })
        result = list_dir.invoke({"path": "/mnt/agent4ml/user-data/workspace"})
        assert "file.txt" in result or "nested" in result

    def test_str_replace_modifies_file(self, tools, sandbox_id) -> None:
        write_file = next(t for t in tools if t.name == "write_file")
        str_replace = next(t for t in tools if t.name == "str_replace")
        read_file = next(t for t in tools if t.name == "read_file")

        write_file.invoke({
            "path": "/mnt/agent4ml/user-data/workspace/replace.txt",
            "content": "old text here",
        })
        str_replace.invoke({
            "path": "/mnt/agent4ml/user-data/workspace/replace.txt",
            "old_str": "old",
            "new_str": "new",
        })
        result = read_file.invoke({"path": "/mnt/agent4ml/user-data/workspace/replace.txt"})
        assert "new text" in result
        assert "old text" not in result

    def test_bash_output_truncation(self, tools, sandbox_id) -> None:
        bash = next(t for t in tools if t.name == "bash")
        result = bash.invoke({"command": "python -c \"print('x' * 15000)\""})
        assert "truncated" in result


class TestMakeSandboxToolsIntegration:
    def test_tools_use_same_provider(self, provider, sandbox_id) -> None:
        tools = make_sandbox_tools(provider)
        bash = next(t for t in tools if t.name == "bash")
        bash.invoke({"command": "echo test"})
        sandbox = provider.get(sandbox_id)
        assert sandbox is not None
