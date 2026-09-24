"""MCP 管理模块层 — 配置化注册、熔断器、fallback、安全层、审计。

公共 API：config / registry / health / loader / audit / 门面 McpManager。
"""
import asyncio
import os

from agent4ml.backend.agents.mcp.audit import McpAuditMiddleware
from agent4ml.backend.agents.mcp.config import (
    McpConfig,
    McpServerConfig,
    load_mcp_config,
    save_mcp_config,
)
from agent4ml.backend.agents.mcp.health import CircuitBreaker
from agent4ml.backend.agents.mcp.loader import McpLoader
from agent4ml.backend.agents.mcp.registry import ToolEntry, ToolRegistry

__all__ = [
    "McpConfig",
    "McpServerConfig",
    "load_mcp_config",
    "save_mcp_config",
    "CircuitBreaker",
    "ToolEntry",
    "ToolRegistry",
    "McpLoader",
    "McpAuditMiddleware",
    "McpManager",
    "build_mcp_manager",
]


class McpManager:
    """MCP 管理门面。聚合 config + registry + loader + audit。

    INVARIANT:
    - bootstrap 构造一次，随 AppRuntime 生命周期
    - load_startup() eager 并行连接，失败不阻塞
    - add_server() 运行时单 server 加载，串行加锁
    - get_tools(groups) 供 agent 注入
    - get_audit_middleware() 供 middleware 链注入
    - shutdown() 清理连接
    - switch_expert_mode 不重建（只调 get_tools 切 group）
    """

    def __init__(self, config: McpConfig) -> None:
        self._config = config
        self._registry = ToolRegistry(config)
        self._loader = McpLoader(config, self._registry)
        self._audit = McpAuditMiddleware(self._registry, self._loader.sanitizer)
        self._add_lock = asyncio.Lock()

    async def load_startup(self) -> None:
        """eager 并行连接所有 enabled server，失败不阻塞。"""
        await self._loader.load_startup()

    async def add_server(self, server_config: McpServerConfig) -> bool:
        """运行时加载单个 MCP server。串行加锁。

        成功 → 注册到 registry + 持久化 + 返 True。
        失败 → logger.error，不改 registry，返 False。
        """
        async with self._add_lock:
            try:
                tools = await self._loader._connect_server(server_config)
            except Exception as exc:
                import logging
                logging.getLogger(__name__).error(
                    "add_server %s failed: %s", server_config.name, exc,
                )
                return False
            self._config.servers[server_config.name] = server_config
            self._persist_server(server_config)
            return True

    def _persist_server(self, server_config: McpServerConfig) -> None:
        """写回 YAML（追加到 servers 段，不覆盖现有）。敏感信息转 ${VAR} 占位。"""
        save_mcp_config(self._config)

    def list_servers(self) -> list[dict]:
        """返回已加载 server 列表 + 状态。供 TUI 面板展示。

        返回 [{name, transport, tool_count, health_state}]。
        """
        result: list[dict] = []
        for name, server in self._config.servers.items():
            tool_count = sum(
                1 for entry in self._registry._entries.values()
                if entry.server_name == name
            )
            health = "healthy"
            for entry in self._registry._entries.values():
                if entry.server_name == name and entry.breaker.state != "closed":
                    health = "unhealthy"
                    break
            result.append({
                "name": name,
                "transport": server.transport,
                "tool_count": tool_count,
                "health_state": health,
            })
        return result

    def get_tools(self, groups: list[str]) -> list:
        """按 group 返回工具列表，供 agent 注入。"""
        return self._registry.get_tools_by_group(groups)

    def get_audit_middleware(self) -> McpAuditMiddleware:
        """供 middleware 链注入。"""
        return self._audit

    @property
    def registry(self) -> ToolRegistry:
        """供外化层取 tool_metadata。"""
        return self._registry

    @property
    def loader(self) -> McpLoader:
        """供 reload_mcp_tools 取连接状态。"""
        return self._loader

    async def shutdown(self) -> None:
        """清理所有连接。"""
        await self._loader.shutdown()


def build_mcp_manager(config_path: str | None = None) -> McpManager | None:
    """从 .env 读开关 + YAML 加载。enabled=false 或无配置返 None。

    读 AGENT4ML_MCP_ENABLED（缺省 false）+ AGENT4ML_MCP_CONFIG_PATH。
    """
    enabled = os.environ.get("AGENT4ML_MCP_ENABLED", "false").lower() == "true"
    if not enabled:
        return None
    config = load_mcp_config(config_path)
    if not config.servers:
        return None
    return McpManager(config)
