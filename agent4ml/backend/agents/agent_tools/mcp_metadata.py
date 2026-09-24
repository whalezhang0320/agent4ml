from __future__ import annotations

from langchain_core.tools import BaseTool

MCP_TOOL_METADATA_KEY = "agent4ml_mcp"


def tag_mcp_tool(tool: BaseTool) -> BaseTool:
    tool.metadata = {**(tool.metadata or {}), MCP_TOOL_METADATA_KEY: True}
    return tool


def is_mcp_tool(tool: BaseTool) -> bool:
    return bool((tool.metadata or {}).get(MCP_TOOL_METADATA_KEY))
