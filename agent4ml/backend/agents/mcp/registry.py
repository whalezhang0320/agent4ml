"""工具注册表 — 注册/发现/namespace/fallback 等价类。

INVARIANT:
- register 按 builtin > mcp > sandbox 优先级去重（先入者保留）
- get(name) 精确查单工具
- get_with_fallback(name) 按 config fallback_chains 查首个 healthy 工具
- get_tools_by_group(groups) 按 core/deferred/sandbox 分组返回
- 线程安全：_lock 保护字典操作
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Callable

from langchain_core.tools import BaseTool

from agent4ml.backend.agents.mcp.health import CircuitBreaker

logger = logging.getLogger(__name__)

_SOURCE_PRIORITY = {"builtin": 0, "mcp": 1, "sandbox": 2}


@dataclass
class ToolEntry:
    """工具注册条目。

    tool: LangChain BaseTool
    source: "builtin" | "mcp" | "sandbox"
    server_name: MCP server 名（builtin/sandbox 为 None）
    metadata: {typical_output_tokens, ...}
    breaker: per-tool CircuitBreaker
    """
    tool: BaseTool
    source: str
    server_name: str | None = None
    metadata: dict = field(default_factory=dict)
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)


class ToolRegistry:
    """工具注册表。config 驱动 fallback_chains + core_tools。

    INVARIANT:
    - register 同名先入者保留，后入者按 source 优先级丢弃（builtin > mcp > sandbox）
    - get_with_fallback 解析 "server:tool" / "builtin:tool" / "sandbox:tool" key
    - get_tools_by_group: core=config.core_tools, sandbox=source=="sandbox", deferred=其余
    """

    def __init__(self, config=None) -> None:
        self._config = config
        self._entries: dict[str, ToolEntry] = {}
        self._lock = threading.Lock()
        self._fallback_chains: dict[str, list[str]] = (
            config.fallback_chains if config else {}
        )
        self._core_tools: set[str] = set(config.core_tools) if config else set()

    def register(self, entry: ToolEntry) -> bool:
        """注册工具。同名先入者保留，后入者丢弃。返 True 表示注册成功。"""
        name = entry.tool.name
        with self._lock:
            existing = self._entries.get(name)
            if existing is None:
                self._entries[name] = entry
                logger.info("registered tool %s (source=%s)", name, entry.source)
                return True
            if _SOURCE_PRIORITY.get(entry.source, 99) < _SOURCE_PRIORITY.get(existing.source, 99):
                self._entries[name] = entry
                logger.warning(
                    "tool %s override: %s → %s (higher priority)",
                    name, existing.source, entry.source,
                )
                return True
            logger.warning(
                "tool %s already registered (%s), skip %s",
                name, existing.source, entry.source,
            )
            return False

    def get(self, name: str) -> ToolEntry | None:
        """精确查单工具。"""
        with self._lock:
            return self._entries.get(name)

    def _resolve_candidate(self, key: str) -> ToolEntry | None:
        """解析 fallback key。

        "server:tool" → 查 server_name 匹配 + tool.name 匹配
        "builtin:tool" → 查 source=="builtin" + tool.name 匹配
        "sandbox:tool" → 查 source=="sandbox" + tool.name 匹配
        纯 "tool" → 直接按 name 查
        """
        if ":" not in key:
            with self._lock:
                return self._entries.get(key)
        prefix, _, tool_name = key.partition(":")
        with self._lock:
            for entry in self._entries.values():
                if entry.tool.name != tool_name:
                    continue
                if prefix == "builtin" and entry.source == "builtin":
                    return entry
                if prefix == "sandbox" and entry.source == "sandbox":
                    return entry
                if entry.server_name == prefix:
                    return entry
        return None

    def get_with_fallback(self, name: str) -> ToolEntry | None:
        """按 fallback_chains 查首个 healthy 工具。

        无 fallback_chains 声明 → 直接 get(name)
        有声明 → 按链顺序查，首个 breaker.allow_call() 的胜出
        全挂 → None
        """
        chain = self._fallback_chains.get(name)
        if not chain:
            entry = self.get(name)
            if entry and entry.breaker.allow_call():
                return entry
            return None
        for key in chain:
            entry = self._resolve_candidate(key)
            if entry and entry.breaker.allow_call():
                return entry
        return None

    def get_tools_by_group(self, groups: list[str]) -> list[BaseTool]:
        """按 group 返回工具列表。

        core: config.core_tools 声明的工具
        sandbox: source=="sandbox"
        deferred: 其余全部
        """
        groups_set = set(groups)
        with self._lock:
            entries = list(self._entries.values())
        result: list[BaseTool] = []
        for entry in entries:
            name = entry.tool.name
            if name in self._core_tools:
                group = "core"
            elif entry.source == "sandbox":
                group = "sandbox"
            else:
                group = "deferred"
            if group in groups_set:
                result.append(entry.tool)
        return result

    def get_metadata(self, name: str) -> dict | None:
        """返回工具元数据。"""
        with self._lock:
            entry = self._entries.get(name)
            return dict(entry.metadata) if entry else None

    def mark_unhealthy(self, tool_name: str) -> None:
        """标记工具不健康（触发 breaker.record_failure）。"""
        with self._lock:
            entry = self._entries.get(tool_name)
        if entry:
            entry.breaker.record_failure()

    def mark_healthy(self, tool_name: str) -> None:
        """标记工具健康（触发 breaker.record_success）。"""
        with self._lock:
            entry = self._entries.get(tool_name)
        if entry:
            entry.breaker.record_success()

    def get_all_metadata(self) -> dict[str, dict]:
        """返回全部工具元数据（供外化层批量取）。"""
        with self._lock:
            return {name: dict(entry.metadata) for name, entry in self._entries.items()}
