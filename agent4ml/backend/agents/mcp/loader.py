"""MCP 连接生命周期 — eager 并行加载 + 按需 refresh + 安全层应用。

INVARIANT:
- load_startup: asyncio.gather 并行连接所有 enabled server，单 server 失败不阻塞
- per server: EnvFilter.check_env → MultiServerMCPClient → get_tools → DescriptionScanner → ToolRegistry.register
- refresh: 仅重连 unhealthy server，增量更新 registry
- shutdown: 清理所有连接
- guards 按顺序应用：EnvFilter → DescriptionScanner
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from agent4ml.backend.agents.mcp.config import McpConfig, McpServerConfig
from agent4ml.backend.agents.mcp.guards import (
    CredentialSanitizer,
    DescriptionScanner,
    EnvFilter,
    SecurityGuard,
)
from agent4ml.backend.agents.mcp.registry import ToolEntry, ToolRegistry

logger = logging.getLogger(__name__)

# SDK transport 映射：config http → SDK streamable_http
_SDK_TRANSPORT_MAP = {"stdio": "stdio", "sse": "sse", "http": "streamable_http"}


def _build_connection(server: McpServerConfig, filtered_env: dict[str, str]) -> dict[str, Any]:
    """McpServerConfig → SDK connection dict（适配 transport 字段名）。"""
    transport = _SDK_TRANSPORT_MAP.get(server.transport, "stdio")
    conn: dict[str, Any] = {"transport": transport}
    if transport == "stdio":
        conn["command"] = server.command or ""
        conn["args"] = list(server.args)
        conn["env"] = filtered_env
    else:
        conn["url"] = server.url or ""
        conn["headers"] = dict(server.headers)
    return conn


class McpLoader:
    """MCP 连接生命周期管理。

    持有 config + registry + guards 列表。
    load_startup 并行连接，refresh 重连失败 server，shutdown 清理。
    """

    def __init__(
        self,
        config: McpConfig,
        registry: ToolRegistry,
        guards: list[SecurityGuard] | None = None,
    ) -> None:
        self._config = config
        self._registry = registry
        self._guards = guards or [
            EnvFilter(),
            DescriptionScanner(),
        ]
        self._sanitizer = CredentialSanitizer()
        self._clients: dict[str, MultiServerMCPClient] = {}
        self._lock = asyncio.Lock()

    def _get_env_guard(self) -> EnvFilter | None:
        """找 EnvFilter guard（env 过滤专用）。"""
        for g in self._guards:
            if isinstance(g, EnvFilter):
                return g
        return None

    def _get_description_guard(self) -> DescriptionScanner | None:
        """找 DescriptionScanner guard（描述扫描专用）。"""
        for g in self._guards:
            if isinstance(g, DescriptionScanner):
                return g
        return None

    async def _connect_server(self, server: McpServerConfig) -> list[BaseTool]:
        """连接单个 server，发现并注册工具。失败 raise，由 load_startup 捕获。"""
        env_guard = self._get_env_guard()
        filtered_env = env_guard.check_env(server.env) if env_guard else dict(server.env)
        connection = _build_connection(server, filtered_env)
        client = MultiServerMCPClient({server.name: connection})
        try:
            tools = await asyncio.wait_for(
                client.get_tools(),
                timeout=server.connect_timeout,
            )
        except asyncio.TimeoutError:
            raise RuntimeError(f"server {server.name} connect timeout ({server.connect_timeout}s)")
        self._clients[server.name] = client
        registered = self._register_tools(server, tools)
        return registered

    def _register_tools(self, server: McpServerConfig, tools: list[BaseTool]) -> list[BaseTool]:
        """注册工具到 registry，应用 include/exclude 过滤 + 描述扫描。"""
        desc_guard = self._get_description_guard()
        include_set = set(server.include_tools) if server.include_tools else None
        exclude_set = set(server.exclude_tools)
        metadata = self._config.tool_metadata
        registered: list[BaseTool] = []
        for tool in tools:
            name = tool.name
            if include_set is not None and name not in include_set:
                logger.info("tool %s not in include list, skip", name)
                continue
            if name in exclude_set:
                logger.info("tool %s in exclude list, skip", name)
                continue
            if desc_guard and desc_guard.scan_description(name, tool.description or ""):
                logger.warning("tool %s description suspicious, skip registration", name)
                continue
            entry = ToolEntry(
                tool=tool,
                source="mcp",
                server_name=server.name,
                metadata=dict(metadata.get(name, {})),
            )
            if self._registry.register(entry):
                registered.append(tool)
        logger.info(
            "server %s: %d/%d tools registered",
            server.name, len(registered), len(tools),
        )
        return registered

    async def load_startup(self) -> list[BaseTool]:
        """并行连接所有 enabled server，单 server 失败不阻塞。

        per server 失败 → logger.error + mark_unhealthy，其余继续。
        """
        enabled_servers = [s for s in self._config.servers.values() if s.enabled]
        if not enabled_servers:
            logger.info("no enabled MCP servers, skip loading")
            return []
        tasks = [self._connect_server(server) for server in enabled_servers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        all_tools: list[BaseTool] = []
        for server, result in zip(enabled_servers, results):
            if isinstance(result, Exception):
                logger.error("server %s failed: %s", server.name, result)
                for tool_name in self._config.tool_metadata:
                    if self._config.tool_metadata[tool_name].get("source") == "mcp":
                        self._registry.mark_unhealthy(tool_name)
                continue
            all_tools.extend(result)
        logger.info(
            "MCP startup complete: %d servers, %d tools total",
            len(enabled_servers), len(all_tools),
        )
        return all_tools

    async def refresh(self) -> None:
        """重连 unhealthy server，增量更新 registry。

        简化实现：重新连接所有 server（registry.register 幂等，同名先入者保留）。
        生产级可优化为仅重连 breaker.state != "closed" 的。
        """
        async with self._lock:
            enabled_servers = [s for s in self._config.servers.values() if s.enabled]
            tasks = [self._connect_server(server) for server in enabled_servers]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for server, result in zip(enabled_servers, results):
                if isinstance(result, Exception):
                    logger.warning("refresh server %s failed: %s", server.name, result)
                else:
                    logger.info("refresh server %s ok: %d tools", server.name, len(result))

    async def shutdown(self) -> None:
        """清理 client 引用。

        ``MultiServerMCPClient.get_tools`` 为发现和每次工具调用各自创建短生命周期
        session；新版 ``langchain-mcp-adapters`` 的 client 不是异步上下文管理器，
        因此这里不应调用 ``__aexit__``。
        """
        self._clients.clear()
        logger.info("MCP loader shutdown complete")

    @property
    def sanitizer(self) -> CredentialSanitizer:
        """暴露 sanitizer 供 audit middleware 用。"""
        return self._sanitizer
