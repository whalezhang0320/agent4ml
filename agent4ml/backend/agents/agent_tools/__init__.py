"""Agent tool loading — unified BaseTool entry points.

MCP 工具加载已迁移至 agent4ml.backend.agents.mcp（McpManager）。
"""

from agent4ml.backend.agents.agent_tools.available import get_available_tools, select_search_tool
from agent4ml.backend.agents.agent_tools.builtin import get_builtin_tools
from agent4ml.backend.agents.agent_tools.mcp_metadata import is_mcp_tool, tag_mcp_tool

__all__ = [
    "get_available_tools",
    "get_builtin_tools",
    "is_mcp_tool",
    "select_search_tool",
    "tag_mcp_tool",
]
