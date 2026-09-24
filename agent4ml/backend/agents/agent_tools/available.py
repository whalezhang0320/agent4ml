from __future__ import annotations

import logging

from langchain_core.tools import BaseTool

from agent4ml.backend.agents.agent_tools.builtin import get_builtin_tools

logger = logging.getLogger(__name__)

# 工具 group 映射（builtin 工具分组标记）。
# core: 基础工具，default + expert 都加载（省上下文）
# deferred: 深度工具，仅 expert 加载
# sandbox: 沙箱工具（bash/read_file/write_file/list_dir/str_replace/present_files）
CORE_TOOL_NAMES: set[str] = {"web_search", "browse_page", "read_snapshot", "skill_search"}
SANDBOX_TOOL_NAMES: set[str] = {"bash", "read_file", "write_file", "list_dir", "str_replace", "present_files"}


def _tool_group(name: str) -> str:
    """工具名 → group。core 白名单内归 core，sandbox 白名单内归 sandbox，其余归 deferred。"""
    if name in CORE_TOOL_NAMES:
        return "core"
    if name in SANDBOX_TOOL_NAMES:
        return "sandbox"
    return "deferred"


def dedupe_by_name(tools: list[BaseTool]) -> list[BaseTool]:
    seen: set[str] = set()
    result: list[BaseTool] = []
    for tool in tools:
        if tool.name not in seen:
            seen.add(tool.name)
            result.append(tool)
    return result


def select_search_tool(tools: list[BaseTool]) -> BaseTool | None:
    for tool in tools:
        if "search" in tool.name.lower():
            return tool
    return None


def get_available_tools(
    groups: list[str] | None = None,
    include_mcp: bool = True,
) -> list[BaseTool]:
    """加载 builtin 工具，可选按 group 过滤。

    MCP 工具由 McpManager 管理（bootstrap 通过 build_mcp_manager 加载），
    此函数只返 builtin 工具（ddg_search + read_snapshot）。
    sandbox 工具由 bootstrap 注入 CapabilityRegistry，不在此加载。

    Args:
        groups: group 白名单。None 加载全部；["core"] 只加载核心工具。
        include_mcp: 保留参数向后兼容，当前无 MCP 加载逻辑（MCP 由 mcp_manager 管）。
    """
    tools = get_builtin_tools()
    all_tools = dedupe_by_name(tools)

    if groups is None:
        return all_tools

    groups_set = set(groups)
    filtered = [t for t in all_tools if _tool_group(t.name) in groups_set]
    skipped = [t.name for t in all_tools if _tool_group(t.name) not in groups_set]
    if skipped:
        logger.info("tools filtered out by groups=%s: %s", groups, skipped)
    return filtered
