"""B11+ /mcp 命令单测（S3）— list 展示 + reload pending + 无 manager 静默。"""
from __future__ import annotations

from io import StringIO

from rich.console import Console

from agent4ml.backend.app.cli.commands import CommandContext, _cmd_mcp, handle_command


class _FakeMcpManager:
    def __init__(self, servers=None):
        self._servers = servers if servers is not None else []

    def list_servers(self):
        return self._servers


def _ctx(state, arg, runtime=None) -> CommandContext:
    return CommandContext(
        console=Console(file=StringIO(), force_terminal=False),
        renderer=None, state=state, runtime=runtime, arg=arg,
    )


def test_mcp_list_with_manager():
    servers = [
        {"name": "tavily", "transport": "stdio", "tool_count": 3, "health_state": "healthy"},
        {"name": "broken", "transport": "stdio", "tool_count": 0, "health_state": "unhealthy"},
    ]
    rt = type("R", (), {"mcp_manager": _FakeMcpManager(servers)})()
    _ctx({}, "list", runtime=rt)  # 不抛
    _cmd_mcp(_ctx({}, "list", runtime=rt))  # 打印 servers


def test_mcp_list_no_manager():
    rt = type("R", (), {"mcp_manager": None})()
    _cmd_mcp(_ctx({}, "list", runtime=rt))  # 提示 not enabled，不抛


def test_mcp_list_empty_servers():
    rt = type("R", (), {"mcp_manager": _FakeMcpManager([])})()
    _cmd_mcp(_ctx({}, "list", runtime=rt))  # 提示 No servers，不抛


def test_mcp_reload_sets_pending():
    state: dict = {}
    rt = type("R", (), {"mcp_manager": _FakeMcpManager()})()
    _cmd_mcp(_ctx(state, "reload", runtime=rt))
    assert state["pending_mcp_reload"] is True


def test_mcp_reload_no_manager_still_sets_pending():
    """reload 不依赖 manager（主循环调 reload_mcp_tools，无 manager 时 graph 重建无 MCP）。"""
    state: dict = {}
    rt = type("R", (), {"mcp_manager": None})()
    _cmd_mcp(_ctx(state, "reload", runtime=rt))
    assert state["pending_mcp_reload"] is True


def test_mcp_no_arg_usage():
    rt = type("R", (), {"mcp_manager": _FakeMcpManager()})()
    _cmd_mcp(_ctx({}, "", runtime=rt))  # 打印 usage，不抛


def test_handle_command_dispatches_mcp_reload():
    state: dict = {}
    console = Console(file=StringIO(), force_terminal=False)
    handle_command("/mcp reload", console, None, state, None)
    assert state["pending_mcp_reload"] is True


def test_handle_command_dispatches_mcp_list():
    rt = type("R", (), {"mcp_manager": _FakeMcpManager([{"name": "x", "transport": "stdio", "tool_count": 1, "health_state": "healthy"}])})()
    console = Console(file=StringIO(), force_terminal=False)
    handle_command("/mcp list", console, None, {}, rt)  # 不抛
