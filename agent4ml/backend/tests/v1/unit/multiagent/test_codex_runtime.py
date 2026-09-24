"""CodexRuntime 单测 — MCP config 构造 + 超时 + crash + ACP mock。"""
from __future__ import annotations

import asyncio

import pytest

from agent4ml.backend.agents.multiagent.exceptions import (
    SpecialistCrashError,
    SpecialistStartupError,
    SpecialistTimeoutError,
)
from agent4ml.backend.agents.multiagent.runtimes.codex_runtime import CodexRuntime
from agent4ml.backend.agents.multiagent.types import SpecialistRequest


def _make_request(**kwargs) -> SpecialistRequest:
    defaults = dict(
        goal="write a function",
        success_criteria="function exists and passes tests",
        context_summary="python project",
        sandbox_id="sb123abc",
        artifacts_path="/workspace",
        timeout_seconds=60,
    )
    defaults.update(kwargs)
    return SpecialistRequest(**defaults)


def test_build_mcp_config_with_sandbox_id():
    rt = CodexRuntime()
    config = rt._build_mcp_config("sb123abc")
    assert len(config) == 1
    server = config[0]
    assert server["name"] == "agent4ml_sandbox"
    assert server["type"] == "stdio"
    assert "--sandbox-id" in server["args"]
    assert "sb123abc" in server["args"]


def test_build_mcp_config_none_sandbox_id():
    rt = CodexRuntime()
    assert rt._build_mcp_config(None) == []


def test_build_env_without_codex_auth_path(monkeypatch):
    """无任何 auth env vars 时返 None（白名单全空）。"""
    # 清除所有 auth-related env vars（Bug C 修复后透传多个 vars）
    for var in ("CODEX_AUTH_PATH", "OPENAI_API_KEY", "CODEX_HOME"):
        monkeypatch.delenv(var, raising=False)
    rt = CodexRuntime()
    assert rt._build_env() is None


def test_build_env_with_codex_auth_path(monkeypatch):
    """CODEX_AUTH_PATH 单独设时只透传该 var。"""
    for var in ("OPENAI_API_KEY", "CODEX_HOME"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CODEX_AUTH_PATH", "/custom/auth.json")
    rt = CodexRuntime()
    env = rt._build_env()
    assert env == {"CODEX_AUTH_PATH": "/custom/auth.json"}


def test_build_env_with_openai_api_key(monkeypatch):
    """OPENAI_API_KEY 透传（Bug C 修复，API key 模式）。"""
    for var in ("CODEX_AUTH_PATH", "CODEX_HOME"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    rt = CodexRuntime()
    env = rt._build_env()
    assert env == {"OPENAI_API_KEY": "sk-test-key"}


def test_build_env_with_codex_home(monkeypatch):
    """CODEX_HOME 透传（Bug C 修复，自定义配置目录）。"""
    for var in ("CODEX_AUTH_PATH", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CODEX_HOME", "/custom/codex/home")
    rt = CodexRuntime()
    env = rt._build_env()
    assert env == {"CODEX_HOME": "/custom/codex/home"}


def test_build_env_with_multiple_auth_vars(monkeypatch):
    """多个 auth vars 同时设时全部透传（白名单聚合）。"""
    monkeypatch.setenv("CODEX_AUTH_PATH", "/custom/auth.json")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    monkeypatch.setenv("CODEX_HOME", "/custom/codex/home")
    rt = CodexRuntime()
    env = rt._build_env()
    assert env == {
        "CODEX_AUTH_PATH": "/custom/auth.json",
        "OPENAI_API_KEY": "sk-test-key",
        "CODEX_HOME": "/custom/codex/home",
    }


def test_invoke_success_via_mock(monkeypatch):
    rt = CodexRuntime()

    async def _mock_run(request):
        return "mock output from codex"

    monkeypatch.setattr(rt, "_run_acp_session", _mock_run)
    result = rt.invoke(_make_request())
    assert result.raw_output == "mock output from codex"
    assert result.duration_seconds >= 0


def test_invoke_timeout(monkeypatch):
    rt = CodexRuntime()

    async def _mock_timeout(request):
        raise asyncio.TimeoutError()

    monkeypatch.setattr(rt, "_run_acp_session", _mock_timeout)
    with pytest.raises(SpecialistTimeoutError):
        rt.invoke(_make_request(timeout_seconds=5))


def test_invoke_crash(monkeypatch):
    rt = CodexRuntime()

    async def _mock_crash(request):
        raise RuntimeError("process crashed")

    monkeypatch.setattr(rt, "_run_acp_session", _mock_crash)
    with pytest.raises(SpecialistCrashError):
        rt.invoke(_make_request())


def test_invoke_file_not_found(monkeypatch):
    rt = CodexRuntime()

    async def _mock_fnf(request):
        raise FileNotFoundError("npx not found")

    monkeypatch.setattr(rt, "_run_acp_session", _mock_fnf)
    with pytest.raises(SpecialistStartupError, match="codex-acp command not found"):
        rt.invoke(_make_request())


def test_invoke_preserves_specialist_error(monkeypatch):
    """SpecialistError 子类不被包装为 CrashError。"""
    rt = CodexRuntime()

    async def _mock_startup(request):
        raise SpecialistStartupError("acp not installed")

    monkeypatch.setattr(rt, "_run_acp_session", _mock_startup)
    with pytest.raises(SpecialistStartupError):
        rt.invoke(_make_request())


def test_custom_command_and_args():
    rt = CodexRuntime(command="node", args=("codex-acp.js",))
    assert rt._command == "node"
    assert rt._args == ("codex-acp.js",)
