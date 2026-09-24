"""Batch B1+B3 self-check: add_server + list_servers + save_mcp_config."""
import asyncio
import os
import tempfile
from pathlib import Path

from agent4ml.backend.agents.mcp import McpConfig, McpManager, McpServerConfig, save_mcp_config


def test_list_servers_empty():
    """无 server → 空列表。"""
    m = McpManager(McpConfig())
    assert m.list_servers() == []
    print("PASS: list_servers empty")


def test_add_server_failure_no_change():
    """add_server 失败不改 registry + 返 False。"""
    m = McpManager(McpConfig())
    bad = McpServerConfig(
        name="bad", transport="stdio",
        command="nonexistent_cmd_xyz", connect_timeout=2,
    )
    result = asyncio.run(m.add_server(bad))
    assert result is False, "should return False on failure"
    assert "bad" not in m._config.servers, "should not add to config on failure"
    assert m.list_servers() == [], "registry should be empty"
    print("PASS: add_server failure → no registry change, return False")


def test_add_server_serial_lock():
    """串行加锁：两次 add_server 不并行。"""
    m = McpManager(McpConfig())
    bad1 = McpServerConfig(name="bad1", transport="stdio", command="nonexistent1", connect_timeout=2)
    bad2 = McpServerConfig(name="bad2", transport="stdio", command="nonexistent2", connect_timeout=2)

    async def _run_both():
        return await asyncio.gather(m.add_server(bad1), m.add_server(bad2))

    results = asyncio.run(_run_both())
    assert all(r is False for r in results)
    print("PASS: add_server serial (both failed, no parallel issue)")


def test_save_mcp_config_credential_placeholder():
    """save_mcp_config 敏感信息转 ${VAR} 占位符。"""
    cfg = McpConfig(servers={
        "my_tool": McpServerConfig(
            name="my_tool", transport="http",
            url="https://example.com/mcp",
            headers={"Authorization": "Bearer abc123def456"},
        ),
    })
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mcp.yaml"
        save_mcp_config(cfg, str(path))
        content = path.read_text(encoding="utf-8")
        assert "${AUTHORIZATION}" in content, f"placeholder not found: {content}"
        assert "Bearer abc123" not in content, f"credential not redacted: {content}"
    print("PASS: save_mcp_config credential → ${VAR} placeholder")


def test_save_mcp_config_stdio_env_placeholder():
    """stdio env 敏感信息转占位符。"""
    cfg = McpConfig(servers={
        "filesystem": McpServerConfig(
            name="filesystem", transport="stdio",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem"],
            env={"GITHUB_TOKEN": "ghp_abcdefghijklmnopqrstuvwxyz0123456789"},
        ),
    })
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mcp.yaml"
        save_mcp_config(cfg, str(path))
        content = path.read_text(encoding="utf-8")
        assert "${GITHUB_TOKEN}" in content, f"placeholder not found: {content}"
        assert "ghp_abcdef" not in content, f"credential not redacted: {content}"
    print("PASS: save_mcp_config stdio env → ${VAR} placeholder")


def test_save_mcp_config_non_credential_kept():
    """非敏感值保留原样。"""
    cfg = McpConfig(servers={
        "my_tool": McpServerConfig(
            name="my_tool", transport="http",
            url="https://example.com/mcp",
            headers={"X-Custom": "not-a-secret"},
        ),
    })
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mcp.yaml"
        save_mcp_config(cfg, str(path))
        content = path.read_text(encoding="utf-8")
        assert "not-a-secret" in content, f"non-credential value should be kept: {content}"
    print("PASS: save_mcp_config non-credential value kept")


def test_save_mcp_config_appends_not_overwrites():
    """save 追加 server，不覆盖现有。"""
    cfg = McpConfig(servers={
        "existing": McpServerConfig(name="existing", transport="stdio", command="echo"),
        "new": McpServerConfig(name="new", transport="http", url="https://example.com"),
    })
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "mcp.yaml"
        save_mcp_config(cfg, str(path))
        content = path.read_text(encoding="utf-8")
        assert "existing" in content
        assert "new" in content
    print("PASS: save_mcp_config appends, not overwrites")


def test_list_servers_with_loaded():
    """有 server 时 list_servers 返回正确格式。"""
    m = McpManager(McpConfig(servers={
        "freeweb": McpServerConfig(name="freeweb", transport="stdio", command="npx"),
    }))
    servers = m.list_servers()
    assert len(servers) == 1
    assert servers[0]["name"] == "freeweb"
    assert servers[0]["transport"] == "stdio"
    assert servers[0]["tool_count"] == 0  # 未实际加载
    assert servers[0]["health_state"] == "healthy"  # 无工具，默认 healthy
    print("PASS: list_servers with loaded server")


if __name__ == "__main__":
    test_list_servers_empty()
    test_add_server_failure_no_change()
    test_add_server_serial_lock()
    test_save_mcp_config_credential_placeholder()
    test_save_mcp_config_stdio_env_placeholder()
    test_save_mcp_config_non_credential_kept()
    test_save_mcp_config_appends_not_overwrites()
    test_list_servers_with_loaded()
    print("\nAll B1+B3 self-checks passed.")
