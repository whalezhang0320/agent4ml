from __future__ import annotations

from langchain_core.tools import BaseTool

from agent4ml.backend.agents.agent_tools.builtin.ask_help import ask_help_tool
from agent4ml.backend.agents.agent_tools.builtin.ddg_search import web_search_tool
from agent4ml.backend.agents.agent_tools.builtin.read_snapshot import read_snapshot
from agent4ml.backend.agents.agent_tools.builtin.skill_search import skill_search


def get_builtin_tools() -> list[BaseTool]:
    """内置工具（优先于 MCP）。dedupe_by_name 优先保留 builtin。

    sandbox 工具（bash/read_file/write_file/list_dir/str_replace）不在此加载——
    由 Stage 4 bootstrap 按需注入：config.sandbox 配了 provider 时调
    make_sandbox_tools(provider) 构造，注入 registry.tools。

    skill_search 归 core group（default + expert 都加载）——让 agent 接到 specialized task
    时主动 check 是否有相关 skill（find-skills 主动触发，设计文档 46 §3.2）。
    """
    return [web_search_tool, read_snapshot, ask_help_tool, skill_search]

