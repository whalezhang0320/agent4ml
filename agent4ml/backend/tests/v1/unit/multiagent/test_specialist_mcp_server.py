"""SpecialistMcpServer 单测 — 8 tool 定义 + sandbox_id 绑定 + 调用 Sandbox 方法 + 错误转 MCP error + main 入口。"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agent4ml.backend.agents.multiagent.mcp.specialist_mcp_server import (
    SpecialistMcpServer,
    _create_sandbox,
    main,
)
from agent4ml.backend.agents.sandbox.exceptions import (
    SandboxCommandError,
    SandboxError,
    SandboxRuntimeError,
)
from agent4ml.backend.agents.sandbox.sandbox import Sandbox
from agent4ml.backend.agents.sandbox.types import GrepMatch


def _mock_sandbox(sandbox_id: str = "sb1") -> MagicMock:
    """构造 mock Sandbox（spec=Sandbox 保证方法签名正确）。"""
    sb = MagicMock(spec=Sandbox)
    sb.id = sandbox_id
    return sb


# ---------------------------------------------------------------------------
# tool 定义
# ---------------------------------------------------------------------------


def test_get_tool_definitions_returns_8():
    sb = _mock_sandbox()
    server = SpecialistMcpServer(sb)
    tools = server.get_tool_definitions()
    assert len(tools) == 8
    names = [t["name"] for t in tools]
    assert names == [
        "bash", "read_file", "write_file", "list_dir",
        "str_replace", "glob", "grep", "download_file",
    ]


def test_tool_definitions_have_required_fields():
    server = SpecialistMcpServer(_mock_sandbox())
    for t in server.get_tool_definitions():
        assert "name" in t
        assert "description" in t
        assert "inputSchema" in t
        assert t["inputSchema"]["type"] == "object"
        assert "properties" in t["inputSchema"]
        assert "required" in t["inputSchema"]


# ---------------------------------------------------------------------------
# sandbox_id 绑定
# ---------------------------------------------------------------------------


def test_sandbox_id_property():
    server = SpecialistMcpServer(_mock_sandbox("my-sandbox"))
    assert server.sandbox_id == "my-sandbox"


# ---------------------------------------------------------------------------
# 调用 Sandbox 方法
# ---------------------------------------------------------------------------


def test_call_bash():
    sb = _mock_sandbox()
    sb.execute_command.return_value = "output"
    server = SpecialistMcpServer(sb)
    assert server.call_tool("bash", {"command": "ls"}) == "output"
    sb.execute_command.assert_called_once_with("ls")


def test_call_read_file():
    sb = _mock_sandbox()
    sb.read_file.return_value = "content"
    server = SpecialistMcpServer(sb)
    assert server.call_tool("read_file", {"path": "/a.py"}) == "content"
    sb.read_file.assert_called_once_with("/a.py")


def test_call_write_file():
    sb = _mock_sandbox()
    server = SpecialistMcpServer(sb)
    result = server.call_tool("write_file", {"path": "/a.py", "content": "hello"})
    assert "wrote 5 chars" in result
    sb.write_file.assert_called_once_with("/a.py", "hello", append=False)


def test_call_write_file_append():
    sb = _mock_sandbox()
    server = SpecialistMcpServer(sb)
    server.call_tool("write_file", {"path": "/a.py", "content": "x", "append": True})
    sb.write_file.assert_called_once_with("/a.py", "x", append=True)


def test_call_list_dir():
    sb = _mock_sandbox()
    sb.list_dir.return_value = ["/a", "/b"]
    server = SpecialistMcpServer(sb)
    result = server.call_tool("list_dir", {"path": "/"})
    assert result == "/a\n/b"
    sb.list_dir.assert_called_once_with("/", max_depth=2)


def test_call_list_dir_empty():
    sb = _mock_sandbox()
    sb.list_dir.return_value = []
    server = SpecialistMcpServer(sb)
    assert server.call_tool("list_dir", {"path": "/"}) == "(empty)"


def test_call_str_replace():
    sb = _mock_sandbox()
    sb.read_file.return_value = "hello world hello"
    server = SpecialistMcpServer(sb)
    result = server.call_tool("str_replace", {
        "path": "/a.py", "old_str": "hello", "new_str": "hi",
    })
    assert "replaced 1" in result
    sb.write_file.assert_called_once_with("/a.py", "hi world hello")


def test_call_str_replace_all():
    sb = _mock_sandbox()
    sb.read_file.return_value = "hello world hello"
    server = SpecialistMcpServer(sb)
    result = server.call_tool("str_replace", {
        "path": "/a.py", "old_str": "hello", "new_str": "hi", "replace_all": True,
    })
    assert "replaced 2" in result
    sb.write_file.assert_called_once_with("/a.py", "hi world hi")


def test_call_str_replace_not_found():
    sb = _mock_sandbox()
    sb.read_file.return_value = "no match here"
    server = SpecialistMcpServer(sb)
    with pytest.raises(SandboxRuntimeError, match="old_str not found"):
        server.call_tool("str_replace", {
            "path": "/a.py", "old_str": "xyz", "new_str": "abc",
        })


def test_call_glob():
    sb = _mock_sandbox()
    sb.glob.return_value = (["/a.py", "/b.py"], False)
    server = SpecialistMcpServer(sb)
    result = server.call_tool("glob", {"path": "/", "pattern": "**/*.py"})
    assert result == "/a.py\n/b.py"


def test_call_glob_truncated():
    sb = _mock_sandbox()
    sb.glob.return_value = (["/a.py"], True)
    server = SpecialistMcpServer(sb)
    result = server.call_tool("glob", {"path": "/", "pattern": "**/*.py"})
    assert "(truncated)" in result


def test_call_glob_empty():
    sb = _mock_sandbox()
    sb.glob.return_value = ([], False)
    server = SpecialistMcpServer(sb)
    assert server.call_tool("glob", {"path": "/", "pattern": "**/*.xyz"}) == "(empty)"


def test_call_grep():
    sb = _mock_sandbox()
    sb.grep.return_value = (
        [GrepMatch(path="/a.py", line_number=10, line="hello world")],
        False,
    )
    server = SpecialistMcpServer(sb)
    result = server.call_tool("grep", {"path": "/", "pattern": "hello"})
    assert "/a.py:10:hello world" in result


def test_call_grep_truncated():
    sb = _mock_sandbox()
    sb.grep.return_value = ([], True)
    server = SpecialistMcpServer(sb)
    result = server.call_tool("grep", {"path": "/", "pattern": "x"})
    assert "(truncated)" in result


def test_call_grep_no_matches():
    sb = _mock_sandbox()
    sb.grep.return_value = ([], False)
    server = SpecialistMcpServer(sb)
    assert server.call_tool("grep", {"path": "/", "pattern": "x"}) == "(no matches)"


def test_call_download_file():
    sb = _mock_sandbox()
    sb.download_file.return_value = b"binary content"
    server = SpecialistMcpServer(sb)
    assert server.call_tool("download_file", {"path": "/a.bin"}) == "binary content"


# ---------------------------------------------------------------------------
# 错误转 MCP error
# ---------------------------------------------------------------------------


def test_sandbox_error_propagates():
    """SandboxError 从 call_tool 抛出（MCP handler 负责转 error response）。"""
    sb = _mock_sandbox()
    sb.execute_command.side_effect = SandboxCommandError("fail", command="bad")
    server = SpecialistMcpServer(sb)
    with pytest.raises(SandboxError):
        server.call_tool("bash", {"command": "bad"})


def test_unknown_tool_raises():
    server = SpecialistMcpServer(_mock_sandbox())
    with pytest.raises(ValueError, match="unknown tool"):
        server.call_tool("nonexistent", {})


# ---------------------------------------------------------------------------
# main 入口参数解析
# ---------------------------------------------------------------------------


def test_main_requires_sandbox_id():
    with pytest.raises(SystemExit):
        main([])


def test_create_local_sandbox():
    """_create_sandbox (无 --sandbox-url) 构造 local Sandbox 实例。"""
    import argparse

    args = argparse.Namespace(
        sandbox_id="test-sb-id", sandbox_url=None, sandbox_root=None
    )
    sandbox = _create_sandbox(args)
    assert isinstance(sandbox, Sandbox)
    assert sandbox.id == "test-sb-id"
