from langchain_core.tools import BaseTool, tool

from agent4ml.backend.agents.agent_tools.available import (
    CORE_TOOL_NAMES,
    _tool_group,
    get_available_tools,
)


class _StubTool(BaseTool):
    """可指定 name 的测试桩工具。"""
    name: str = "stub"
    description: str = "stub tool"

    def _run(self) -> str:
        return "stub"


def _make_tool(name: str) -> BaseTool:
    t = _StubTool()
    t.name = name
    return t


def test_tool_group_core() -> None:
    assert _tool_group("web_search") == "core"
    assert _tool_group("browse_page") == "core"


def test_tool_group_deferred() -> None:
    assert _tool_group("deep_search") == "deferred"
    assert _tool_group("get_page_links") == "deferred"
    assert _tool_group("unknown_tool") == "deferred"


def test_core_tool_names_contains_basics() -> None:
    assert "web_search" in CORE_TOOL_NAMES
    assert "browse_page" in CORE_TOOL_NAMES


def test_get_available_tools_groups_none_returns_all(monkeypatch) -> None:
    """get_available_tools(groups=None) 返全部 builtin。"""
    monkeypatch.setattr(
        "agent4ml.backend.agents.agent_tools.available.get_builtin_tools",
        lambda: [_make_tool("web_search"), _make_tool("deep_search")],
    )
    result = get_available_tools(groups=None)
    assert {t.name for t in result} == {"web_search", "deep_search"}


def test_get_available_tools_groups_core_filters(monkeypatch) -> None:
    """groups=["core"] 只返 core 白名单内的 builtin。"""
    monkeypatch.setattr(
        "agent4ml.backend.agents.agent_tools.available.get_builtin_tools",
        lambda: [_make_tool("web_search"), _make_tool("deep_search")],
    )
    result = get_available_tools(groups=["core"])
    assert {t.name for t in result} == {"web_search"}


def test_get_available_tools_groups_core_and_deferred_returns_all(monkeypatch) -> None:
    """groups=["core","deferred"] 返 core + deferred builtin。"""
    monkeypatch.setattr(
        "agent4ml.backend.agents.agent_tools.available.get_builtin_tools",
        lambda: [_make_tool("web_search"), _make_tool("deep_search")],
    )
    result = get_available_tools(groups=["core", "deferred"])
    assert {t.name for t in result} == {"web_search", "deep_search"}


def test_get_available_tools_no_mcp_loading(monkeypatch) -> None:
    """get_available_tools 不再加载 MCP（include_mcp 参数保留但无 MCP 逻辑）。"""
    monkeypatch.setattr(
        "agent4ml.backend.agents.agent_tools.available.get_builtin_tools",
        lambda: [_make_tool("web_search")],
    )
    result = get_available_tools(groups=None, include_mcp=True)
    assert {t.name for t in result} == {"web_search"}
