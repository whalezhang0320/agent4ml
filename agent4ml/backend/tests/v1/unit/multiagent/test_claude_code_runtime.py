"""ClaudeCodeRuntime 单测 — CLI 命令构造 + MCP 配置 + 超时 + crash + subprocess mock。"""
from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from agent4ml.backend.agents.multiagent.exceptions import (
    SpecialistCrashError,
    SpecialistStartupError,
    SpecialistTimeoutError,
)
from agent4ml.backend.agents.multiagent.runtimes.claude_code_runtime import (
    ClaudeCodeRuntime,
)
from agent4ml.backend.agents.multiagent.types import SpecialistRequest


def _make_request(**kwargs) -> SpecialistRequest:
    defaults = dict(
        goal="review this code",
        success_criteria="review comments provided",
        context_summary="python project",
        sandbox_id="sb456def",
        artifacts_path="/workspace",
        timeout_seconds=60,
    )
    defaults.update(kwargs)
    return SpecialistRequest(**defaults)


def test_build_command():
    rt = ClaudeCodeRuntime()
    cmd = rt._build_command(_make_request(goal="hello world"))
    assert cmd == ["claude", "--print", "hello world"]


def test_build_mcp_add_command():
    rt = ClaudeCodeRuntime()
    cmd = rt._build_mcp_add_command("sb123")
    assert cmd[0] == "claude"
    assert "mcp" in cmd
    assert "add" in cmd
    assert "agent4ml_sandbox" in cmd
    assert "--sandbox-id" in cmd
    assert "sb123" in cmd


def test_invoke_success():
    rt = ClaudeCodeRuntime()
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "review output"

    with patch("subprocess.run", return_value=mock_result):
        result = rt.invoke(_make_request())

    assert result.raw_output == "review output"
    assert result.duration_seconds >= 0


def test_invoke_timeout():
    rt = ClaudeCodeRuntime()

    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=5)):
        with pytest.raises(SpecialistTimeoutError):
            rt.invoke(_make_request(timeout_seconds=5))


def test_invoke_nonzero_exit():
    rt = ClaudeCodeRuntime()
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "error"

    with patch("subprocess.run", return_value=mock_result):
        with pytest.raises(SpecialistCrashError) as exc_info:
            rt.invoke(_make_request())

    assert exc_info.value.details.get("exit_code") == 1


def test_invoke_command_not_found():
    rt = ClaudeCodeRuntime()

    with patch("subprocess.run", side_effect=FileNotFoundError("claude not found")):
        with pytest.raises(SpecialistStartupError, match="claude command not found"):
            rt.invoke(_make_request())


def test_custom_command():
    rt = ClaudeCodeRuntime(command="/usr/local/bin/claude")
    assert rt._command == "/usr/local/bin/claude"


def test_configure_mcp_runs_subprocess():
    rt = ClaudeCodeRuntime()
    mock_result = MagicMock()
    mock_result.returncode = 0

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        rt.configure_mcp("sb789")

    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert "mcp" in cmd
    assert "add" in cmd


# ---------------------------------------------------------------------------
# Bug C 修复（设计文档 46 §4.3）：_build_env 透传 auth vars
# ---------------------------------------------------------------------------


def test_build_env_without_auth_vars(monkeypatch):
    """无任何 auth env vars 时返 None。"""
    for var in (
        "CLAUDE_CODE_OAUTH_TOKEN",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_API_KEY",
        "CLAUDE_CODE_CREDENTIALS_PATH",
    ):
        monkeypatch.delenv(var, raising=False)
    rt = ClaudeCodeRuntime()
    assert rt._build_env() is None


def test_build_env_with_oauth_token(monkeypatch):
    """CLAUDE_CODE_OAUTH_TOKEN 透传（Bug C 修复）。"""
    for var in (
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_API_KEY",
        "CLAUDE_CODE_CREDENTIALS_PATH",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth-token-123")
    rt = ClaudeCodeRuntime()
    env = rt._build_env()
    # _build_env 返 merge 后的 env（父 env + auth vars），验证 auth var 在其中
    assert env is not None
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "oauth-token-123"


def test_build_env_with_anthropic_auth_token(monkeypatch):
    """ANTHROPIC_AUTH_TOKEN 透传。"""
    for var in (
        "CLAUDE_CODE_OAUTH_TOKEN",
        "ANTHROPIC_API_KEY",
        "CLAUDE_CODE_CREDENTIALS_PATH",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "auth-token-456")
    rt = ClaudeCodeRuntime()
    env = rt._build_env()
    assert env is not None
    assert env["ANTHROPIC_AUTH_TOKEN"] == "auth-token-456"


def test_build_env_with_anthropic_api_key(monkeypatch):
    """ANTHROPIC_API_KEY 透传。"""
    for var in (
        "CLAUDE_CODE_OAUTH_TOKEN",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_CREDENTIALS_PATH",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-789")
    rt = ClaudeCodeRuntime()
    env = rt._build_env()
    assert env is not None
    assert env["ANTHROPIC_API_KEY"] == "sk-ant-789"


def test_build_env_with_credentials_path(monkeypatch):
    """CLAUDE_CODE_CREDENTIALS_PATH 透传。"""
    for var in (
        "CLAUDE_CODE_OAUTH_TOKEN",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CLAUDE_CODE_CREDENTIALS_PATH", "/custom/credentials.json")
    rt = ClaudeCodeRuntime()
    env = rt._build_env()
    assert env is not None
    assert env["CLAUDE_CODE_CREDENTIALS_PATH"] == "/custom/credentials.json"


def test_build_env_merges_parent_env(monkeypatch):
    """_build_env 返 merge 后的 env（父 env + auth vars 覆盖），保证 PATH/HOME 可用。"""
    # 设一个 auth var 触发 merge
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    # 确保父 env 的 PATH 在结果中
    import os
    parent_path = os.environ.get("PATH", "")
    rt = ClaudeCodeRuntime()
    env = rt._build_env()
    assert env is not None
    assert env["PATH"] == parent_path
    assert env["ANTHROPIC_API_KEY"] == "sk-test"


def test_invoke_passes_env_to_subprocess(monkeypatch):
    """invoke 调 subprocess.run 时传 env 参数（Bug C 修复）。"""
    # 设一个 auth var 让 _build_env 返非 None
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    rt = ClaudeCodeRuntime()
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "ok"

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        rt.invoke(_make_request())

    # 验证 subprocess.run 收到 env 参数（不是 None）
    _, kwargs = mock_run.call_args
    assert "env" in kwargs
    assert kwargs["env"] is not None
    assert kwargs["env"]["ANTHROPIC_API_KEY"] == "sk-test"
