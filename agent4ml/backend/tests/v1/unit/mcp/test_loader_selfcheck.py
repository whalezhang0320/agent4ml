"""Batch B5 self-check: McpLoader (no real server, just structural)."""
import asyncio
from agent4ml.backend.agents.mcp.config import McpConfig, McpServerConfig
from agent4ml.backend.agents.mcp.loader import McpLoader
from agent4ml.backend.agents.mcp.registry import ToolRegistry


def test_empty_config():
    """空 config → load_startup 返空列表，不抛。"""
    cfg = McpConfig()
    reg = ToolRegistry(cfg)
    loader = McpLoader(cfg, reg)
    tools = asyncio.run(loader.load_startup())
    assert tools == [], f"expected empty, got {tools}"
    print("PASS: empty config → no tools, no error")


def test_disabled_server_skipped():
    """enabled=false 的 server 被跳过。"""
    cfg = McpConfig(servers={
        "test": McpServerConfig(name="test", transport="stdio", command="echo", enabled=False),
    })
    reg = ToolRegistry(cfg)
    loader = McpLoader(cfg, reg)
    tools = asyncio.run(loader.load_startup())
    assert tools == [], "disabled server should be skipped"
    print("PASS: disabled server skipped")


def test_failed_server_not_blocking():
    """单 server 连接失败不阻塞（command 不存在会失败）。"""
    cfg = McpConfig(servers={
        "bad": McpServerConfig(
            name="bad", transport="stdio", command="nonexistent_cmd_xyz",
            connect_timeout=2,
        ),
    })
    reg = ToolRegistry(cfg)
    loader = McpLoader(cfg, reg)
    tools = asyncio.run(loader.load_startup())
    # 失败 server 不返工具，但也不抛
    assert tools == [], "failed server should not return tools"
    print("PASS: failed server not blocking")


def test_shutdown_safe():
    """shutdown 无 server 时安全。"""
    cfg = McpConfig()
    reg = ToolRegistry(cfg)
    loader = McpLoader(cfg, reg)
    asyncio.run(loader.shutdown())
    print("PASS: shutdown safe (no clients)")


def test_shutdown_clears_clients_without_context_manager() -> None:
    """新版 MultiServerMCPClient 不是 context manager，shutdown 只清引用。"""
    cfg = McpConfig()
    reg = ToolRegistry(cfg)
    loader = McpLoader(cfg, reg)
    loader._clients["test"] = object()  # type: ignore[assignment]

    asyncio.run(loader.shutdown())

    assert loader._clients == {}


def test_env_filter_applied():
    """EnvFilter guard 在 guards 列表内。"""
    cfg = McpConfig()
    reg = ToolRegistry(cfg)
    loader = McpLoader(cfg, reg)
    assert any(type(g).__name__ == "EnvFilter" for g in loader._guards)
    print("PASS: EnvFilter in guards")


def test_sanitizer_exposed():
    """sanitizer 属性可访问。"""
    cfg = McpConfig()
    reg = ToolRegistry(cfg)
    loader = McpLoader(cfg, reg)
    assert loader.sanitizer is not None
    assert loader.sanitizer.sanitize_error("ghp_test") == loader.sanitizer.sanitize_error("ghp_test")
    print("PASS: sanitizer exposed")


if __name__ == "__main__":
    test_empty_config()
    test_disabled_server_skipped()
    test_failed_server_not_blocking()
    test_shutdown_safe()
    test_env_filter_applied()
    test_sanitizer_exposed()
    print("\nAll B5 self-checks passed.")
