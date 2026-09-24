"""SpecialistMcpServer — 暴露 Agent4ML 8 个沙箱接口给 specialist via MCP（stdio）。

设计（spec.md SpecialistMcpServer Requirement + design.md §5）:
- 8 个 MCP tool：bash/read_file/write_file/list_dir/str_replace/glob/grep/download_file
- 直接映射既有 Sandbox 类方法（不重新实现沙箱逻辑，INV#9）
- 经过 PathTranslator + SecurityGuard（既有安全层）
- per-specialist-call 生命周期（每次启动 + 完成关闭）
- sandbox_id 通过 --sandbox-id 命令行参数传递
- 独立入口：python -m agent4ml.backend.agents.multiagent.mcp.specialist_mcp_server --sandbox-id {id}
- 错误转 MCP error response（SandboxError → error text content）
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

from agent4ml.backend.agents.sandbox.exceptions import SandboxError, SandboxRuntimeError
from agent4ml.backend.agents.sandbox.sandbox import Sandbox


def _tool_definitions() -> list[dict[str, Any]]:
    """8 个 MCP tool 定义（name + description + inputSchema）。"""
    return [
        {
            "name": "bash",
            "description": "Execute a bash command in the sandbox.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command to execute."},
                },
                "required": ["command"],
            },
        },
        {
            "name": "read_file",
            "description": "Read the contents of a file.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Virtual path to the file."},
                },
                "required": ["path"],
            },
        },
        {
            "name": "write_file",
            "description": "Write content to a file. Creates parent directories if needed.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Virtual path to the file."},
                    "content": {"type": "string", "description": "Text content to write."},
                    "append": {"type": "boolean", "description": "Append if true, overwrite if false.", "default": False},
                },
                "required": ["path", "content"],
            },
        },
        {
            "name": "list_dir",
            "description": "List directory contents in tree format.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Virtual path to the directory."},
                    "max_depth": {"type": "integer", "description": "Maximum depth to traverse.", "default": 2},
                },
                "required": ["path"],
            },
        },
        {
            "name": "str_replace",
            "description": "Replace text in a file. Reads, replaces, writes back.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Virtual path to the file."},
                    "old_str": {"type": "string", "description": "Text to find."},
                    "new_str": {"type": "string", "description": "Replacement text."},
                    "replace_all": {"type": "boolean", "description": "Replace all occurrences if true.", "default": False},
                },
                "required": ["path", "old_str", "new_str"],
            },
        },
        {
            "name": "glob",
            "description": "Find files matching a glob pattern.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Virtual path to search in."},
                    "pattern": {"type": "string", "description": "Glob pattern (e.g. **/*.py)."},
                    "include_dirs": {"type": "boolean", "description": "Include directories in results.", "default": False},
                    "max_results": {"type": "integer", "description": "Maximum results.", "default": 200},
                },
                "required": ["path", "pattern"],
            },
        },
        {
            "name": "grep",
            "description": "Search for pattern in files.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Virtual path to search in."},
                    "pattern": {"type": "string", "description": "Search pattern (regex unless literal=true)."},
                    "glob": {"type": "string", "description": "File pattern filter (e.g. *.py)."},
                    "literal": {"type": "boolean", "description": "Treat pattern as literal string.", "default": False},
                    "case_sensitive": {"type": "boolean", "description": "Case-sensitive search.", "default": False},
                    "max_results": {"type": "integer", "description": "Maximum results.", "default": 100},
                },
                "required": ["path", "pattern"],
            },
        },
        {
            "name": "download_file",
            "description": "Download a file as bytes (decoded to text).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Virtual path to the file."},
                },
                "required": ["path"],
            },
        },
    ]


class SpecialistMcpServer:
    """暴露 Agent4ML 8 个沙箱接口给 specialist via MCP（stdio）。

    不重新实现沙箱逻辑——调用既有 Sandbox 类方法（经过 PathTranslator + SecurityGuard）。
    生命周期 per-specialist-call（每次启动 + 完成关闭）。
    """

    def __init__(self, sandbox: Sandbox) -> None:
        self._sandbox = sandbox

    @property
    def sandbox_id(self) -> str:
        return self._sandbox.id

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """返 8 个 MCP tool 定义。"""
        return _tool_definitions()

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """dispatch tool call to Sandbox method。

        抛 SandboxError 时由 MCP handler 转 error response。
        抛 ValueError 表示未知 tool name。
        """
        if name == "bash":
            return self._sandbox.execute_command(arguments["command"])

        if name == "read_file":
            return self._sandbox.read_file(arguments["path"])

        if name == "write_file":
            self._sandbox.write_file(
                arguments["path"],
                arguments["content"],
                append=arguments.get("append", False),
            )
            return f"wrote {len(arguments['content'])} chars to {arguments['path']}"

        if name == "list_dir":
            entries = self._sandbox.list_dir(
                arguments["path"],
                max_depth=arguments.get("max_depth", 2),
            )
            return "\n".join(entries) if entries else "(empty)"

        if name == "str_replace":
            return self._str_replace(
                arguments["path"],
                arguments["old_str"],
                arguments["new_str"],
                replace_all=arguments.get("replace_all", False),
            )

        if name == "glob":
            results, truncated = self._sandbox.glob(
                arguments["path"],
                arguments["pattern"],
                include_dirs=arguments.get("include_dirs", False),
                max_results=arguments.get("max_results", 200),
            )
            suffix = " (truncated)" if truncated else ""
            return "\n".join(results) + suffix if results else "(empty)"

        if name == "grep":
            results, truncated = self._sandbox.grep(
                arguments["path"],
                arguments["pattern"],
                glob=arguments.get("glob"),
                literal=arguments.get("literal", False),
                case_sensitive=arguments.get("case_sensitive", False),
                max_results=arguments.get("max_results", 100),
            )
            lines = [f"{r.path}:{r.line_number}:{r.line}" for r in results]
            if truncated:
                lines.append("(truncated)")
            return "\n".join(lines) if lines else "(no matches)"

        if name == "download_file":
            data = self._sandbox.download_file(arguments["path"])
            return data.decode("utf-8", errors="replace")

        raise ValueError(f"unknown tool: {name}")

    def _str_replace(
        self, path: str, old_str: str, new_str: str, replace_all: bool = False
    ) -> str:
        """str_replace 复合操作：read → replace → write（参考 sandbox/integration/tools.py）。"""
        content = self._sandbox.read_file(path)
        if old_str not in content:
            raise SandboxRuntimeError(f"old_str not found in {path}")
        count = content.count(old_str)
        if replace_all:
            new_content = content.replace(old_str, new_str)
        else:
            new_content = content.replace(old_str, new_str, 1)
        self._sandbox.write_file(path, new_content)
        return f"replaced {count if replace_all else 1} occurrence(s) in {path}"

    async def run(self) -> None:
        """stdio MCP server loop（使用 mcp 包）。"""
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import TextContent, Tool

        server = Server("agent4ml-sandbox")

        @server.list_tools()
        async def list_tools() -> list[Tool]:
            return [
                Tool(
                    name=t["name"],
                    description=t["description"],
                    inputSchema=t["inputSchema"],
                )
                for t in self.get_tool_definitions()
            ]

        @server.call_tool()
        async def handle_call_tool(
            name: str, arguments: dict[str, Any] | None
        ) -> list[TextContent]:
            try:
                result = self.call_tool(name, arguments or {})
                return [TextContent(type="text", text=result)]
            except SandboxError as e:
                return [TextContent(type="text", text=f"Error: {e}")]
            except Exception as e:
                return [TextContent(type="text", text=f"Error: {e}")]

        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )


def _create_sandbox(args: argparse.Namespace) -> Sandbox:
    """根据 args 选 runtime:有 --sandbox-url 用 DockerRuntime 连 lead 容器,否则 fallback Local。

    --sandbox-url 有(块 D3):DockerRuntime + DockerPathTranslator + DockerPathGuard
        → specialist 连 lead Docker 容器,写入落同一挂载区。
    --sandbox-url 无:fallback LocalRuntime + LocalPathTranslator + LocalSecurityGuard
        → LocalSandboxProvider 场景(现状)。
    """
    from agent4ml.backend.agents.sandbox.guards.audit_guard import AuditGuard
    from agent4ml.backend.agents.sandbox.sandbox import Sandbox

    if args.sandbox_url:
        from agent4ml.backend.agents.sandbox.guards.docker_path_guard import (
            DockerPathGuard,
        )
        from agent4ml.backend.agents.sandbox.runtimes.docker_runtime import DockerRuntime
        from agent4ml.backend.agents.sandbox.translators.docker_path_translator import (
            DockerPathTranslator,
        )
        runtime = DockerRuntime(args.sandbox_url)
        translator = DockerPathTranslator(args.sandbox_root, args.sandbox_id)
        guard = AuditGuard(DockerPathGuard())
    else:
        from agent4ml.backend.agents.sandbox.guards.local_security_guard import (
            LocalSecurityGuard,
        )
        from agent4ml.backend.agents.sandbox.runtimes.local_runtime import LocalRuntime
        from agent4ml.backend.agents.sandbox.translators.local_path_translator import (
            LocalPathTranslator,
        )
        runtime = LocalRuntime(allow_host_bash=True)
        translator = LocalPathTranslator([])
        guard = AuditGuard(LocalSecurityGuard([]))
    return Sandbox(args.sandbox_id, runtime, translator, guard)


def main(argv: list[str] | None = None) -> None:
    """独立入口:python -m ...specialist_mcp_server --sandbox-id {id} [--sandbox-url URL]"""
    parser = argparse.ArgumentParser(description="Agent4ML Specialist MCP Server")
    parser.add_argument("--sandbox-id", required=True, help="Sandbox ID to bind")
    parser.add_argument(
        "--sandbox-url", default=None, help="Docker sandbox URL (lead container)"
    )
    parser.add_argument(
        "--sandbox-root",
        default=None,
        help="Docker sandbox host root (for reverse_translate)",
    )
    args = parser.parse_args(argv)

    sandbox = _create_sandbox(args)
    server = SpecialistMcpServer(sandbox)
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
