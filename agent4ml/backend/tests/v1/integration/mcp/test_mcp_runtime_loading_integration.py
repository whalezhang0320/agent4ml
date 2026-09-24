"""Batch B6 integration tests: add_server + list_servers + save_mcp_config + reload + panel.

覆盖任务 6.1-6.6（部分用 B1 自检的 test_runtime_loading_selfcheck 补充）。
"""
import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.tools import tool

from agent4ml.backend.agents.mcp import (
    McpConfig,
    McpManager,
    McpServerConfig,
    save_mcp_config,
)


@tool("mock_search")
def mock_search(query: str) -> str:
    """search web"""
    return "result"


# ── 6.1 add_server：成功 + 失败不改 registry + 串行 ──


def test_add_server_failure_no_registry_change():
    """add_server 失败不改 registry + 返 False。"""
    m = McpManager(McpConfig())
    bad = McpServerConfig(
        name="bad", transport="stdio",
        command="nonexistent_cmd_xyz", connect_timeout=2,
    )
    result = asyncio.run(m.add_server(bad))
    assert result is False
    assert "bad" not in m._config.servers
    assert m.list_servers() == []


def test_add_server_serial_lock():
    """串行加锁：两次 add_server 不并行。"""
    m = McpManager(McpConfig())
    bad1 = McpServerConfig(name="bad1", transport="stdio", command="nonexistent1", connect_timeout=2)
    bad2 = McpServerConfig(name="bad2", transport="stdio", command="nonexistent2", connect_timeout=2)

    async def _run_both():
        return await asyncio.gather(m.add_server(bad1), m.add_server(bad2))

    results = asyncio.run(_run_both())
    assert all(r is False for r in results)


# ── 6.2 list_servers：返回格式 + 空 server ──


def test_list_servers_empty():
    """无 server → 空列表。"""
    m = McpManager(McpConfig())
    assert m.list_servers() == []


def test_list_servers_format():
    """有 server 时返回正确格式。"""
    m = McpManager(McpConfig(servers={
        "freeweb": McpServerConfig(name="freeweb", transport="stdio", command="npx"),
    }))
    servers = m.list_servers()
    assert len(servers) == 1
    s = servers[0]
    assert s["name"] == "freeweb"
    assert s["transport"] == "stdio"
    assert "tool_count" in s
    assert "health_state" in s


# ── 6.3 save_mcp_config：YAML 写回 + 敏感信息占位 ──


def test_save_mcp_config_credential_placeholder():
    """敏感信息转 ${VAR} 占位符。"""
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
        assert "${AUTHORIZATION}" in content
        assert "Bearer abc123" not in content


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
        assert "not-a-secret" in content


def test_save_mcp_config_appends():
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


# ── 6.4 reload_mcp_tools：graph 重建（mock） ──


def test_reload_mcp_tools_exists():
    """AppRuntime 有 reload_mcp_tools 方法。"""
    from agent4ml.backend.app.bootstrap import AppRuntime
    assert hasattr(AppRuntime, "reload_mcp_tools")


def test_reload_mcp_tools_preserves_thread_id():
    """reload_mcp_tools 保留 thread_id（mock leader_agent 重建）。"""
    from agent4ml.backend.app.bootstrap import AppRuntime
    from agent4ml.backend.agents.capabilities.registry import CapabilityRegistry
    from agent4ml.backend.agents.journal.run_journal import RunJournal

    with patch("agent4ml.backend.app.bootstrap.make_lead_agent") as mock_make:
        mock_make.return_value = MagicMock()
        runtime = AppRuntime(
            config=MagicMock(),
            capability_registry=MagicMock(spec=CapabilityRegistry),
            run_manager=MagicMock(),
            researcher_model_name="test",
            thread_id="thread-123",
            thread_dir=Path("/tmp"),
            thread_journal=MagicMock(spec=RunJournal),
            leader_agent=MagicMock(),
            mcp_manager=None,
            artifact_server=None,
        )
        new_runtime = runtime.reload_mcp_tools()
        assert new_runtime.thread_id == "thread-123"
        assert mock_make.called


# ── 6.5 McpPanel：字段解析 + 校验（mock add_server） ──


def test_mcp_panel_import():
    """McpPanel 可导入。"""
    from agent4ml.backend.app.tui.mcp_panel import McpPanel
    assert McpPanel is not None


def test_parse_headers_env():
    """_parse_headers_env 解析 key=val,key=val。"""
    from agent4ml.backend.app.tui.mcp_panel import McpPanel
    mcp_mgr = MagicMock(spec=McpManager)
    mcp_mgr._config = McpConfig()
    panel = McpPanel(mcp_mgr)
    result = panel._parse_headers_env("Authorization=Bearer xxx, X-Custom=val")
    assert result == {"Authorization": "Bearer xxx", "X-Custom": "val"}


def test_parse_headers_env_empty():
    """空输入返空 dict。"""
    from agent4ml.backend.app.tui.mcp_panel import McpPanel
    mcp_mgr = MagicMock(spec=McpManager)
    mcp_mgr._config = McpConfig()
    panel = McpPanel(mcp_mgr)
    assert panel._parse_headers_env("") == {}


def test_validate_url_format_blocking():
    """URL 格式非法阻断。"""
    from agent4ml.backend.app.tui.mcp_panel import McpPanel
    mcp_mgr = MagicMock(spec=McpManager)
    mcp_mgr._config = McpConfig()
    panel = McpPanel(mcp_mgr)
    # mock query_one 返回测试值
    with patch.object(panel, "query_one") as mock_q:
        mock_q.side_effect = lambda selector, *args, **kw: MagicMock(value=_get_input_value(selector))
        config, error = panel._validate_and_build_config()
        assert config is None
        assert "http://" in error or "https://" in error


def _get_input_value(selector: str) -> str:
    """辅助：返回测试输入值。"""
    if "transport" in selector:
        return "http"
    if "url" in selector:
        return "ftp://bad.url"
    if "name" in selector:
        return "test_tool"
    return ""


def test_validate_name_dup_blocking():
    """Name 重名阻断。"""
    from agent4ml.backend.app.tui.mcp_panel import McpPanel
    cfg = McpConfig(servers={
        "existing": McpServerConfig(name="existing", transport="http", url="https://example.com"),
    })
    mcp_mgr = MagicMock(spec=McpManager)
    mcp_mgr._config = cfg
    panel = McpPanel(mcp_mgr)
    with patch.object(panel, "query_one") as mock_q:
        mock_q.side_effect = lambda selector, *args, **kw: MagicMock(value=_get_name_dup_value(selector))
        config, error = panel._validate_and_build_config()
        assert config is None
        assert "已存在" in error


def _get_name_dup_value(selector: str) -> str:
    if "transport" in selector:
        return "http"
    if "url" in selector:
        return "https://example.com/mcp"
    if "name" in selector:
        return "existing"
    return ""


# ── 6.6 集成测试：端到端流程（mock） ──


def test_end_to_end_add_server_persist():
    """add_server 成功 → config 更新 + 持久化（mock _connect_server）。"""
    from agent4ml.backend.agents.mcp.registry import ToolEntry
    m = McpManager(McpConfig())
    server = McpServerConfig(
        name="test_tool", transport="http",
        url="https://example.com/mcp",
    )
    # mock _connect_server 成功（返回工具列表 + 手动注册到 registry）
    async def _fake_connect(srv):
        m._registry.register(ToolEntry(tool=mock_search, source="mcp", server_name=srv.name))
        return [mock_search]
    with patch.object(m._loader, "_connect_server", new=_fake_connect):
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["AGENT4ML_MCP_CONFIG_PATH"] = str(Path(tmp) / "mcp.yaml")
            result = asyncio.run(m.add_server(server))
            assert result is True
            assert "test_tool" in m._config.servers
            servers = m.list_servers()
            assert len(servers) == 1
            assert servers[0]["name"] == "test_tool"
            assert servers[0]["tool_count"] == 1


def test_end_to_end_failure_no_persist():
    """add_server 失败 → config 不更新 + 不持久化。"""
    m = McpManager(McpConfig())
    server = McpServerConfig(
        name="bad", transport="stdio",
        command="nonexistent", connect_timeout=2,
    )
    result = asyncio.run(m.add_server(server))
    assert result is False
    assert "bad" not in m._config.servers
