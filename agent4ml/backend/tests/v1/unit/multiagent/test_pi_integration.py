"""Pi specialist 端到端集成测试（P7）。

验证（tasks.md 7.1）：
- test_end_to_end_pi_specialist: mock pi RPC 子进程 + SpecialistMcpServer + 沙箱，验证端到端流程
- test_pi_specialist_fails_returns_error_toolmessage: pi 崩溃，返 error ToolMessage
- test_pi_uses_shared_sandbox: pi 操作的文件 lead agent 立即可见（物理无隔离，决策 1）
- test_pi_no_bypass_security_guard: pi extension 所有操作走 SpecialistMcpServer
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agent4ml.backend.agents.multiagent.exceptions import SpecialistCrashError
from agent4ml.backend.agents.multiagent.runtimes.pi_runtime import PiRuntime
from agent4ml.backend.agents.multiagent.specialists.pi_specialist import PiSpecialist
from agent4ml.backend.agents.multiagent.types import SpecialistRequest


def _make_request(**kwargs) -> SpecialistRequest:
    defaults = dict(
        goal="write auth.py",
        success_criteria="auth.py exists + tests pass",
        context_summary="python project",
        sandbox_id="sb123abc",
        artifacts_path="/workspace",
        timeout_seconds=60,
    )
    defaults.update(kwargs)
    return SpecialistRequest(**defaults)


def _make_event_line(event_dict: dict) -> str:
    return json.dumps(event_dict)


def _mock_proc(stdout_lines: list[str], returncode: int = 0):
    mock_proc = MagicMock()
    mock_proc.stdout = iter(stdout_lines)
    mock_proc.stderr = iter([])
    mock_proc.stdin = MagicMock()
    mock_proc.wait.return_value = returncode
    mock_proc.terminate.return_value = None
    mock_proc.kill.return_value = None
    return mock_proc


def test_end_to_end_pi_specialist():
    """端到端：PiSpecialist.invoke → PiRuntime.invoke → pi RPC 子进程 → 返 SpecialistRawResult。

    mock pi RPC 子进程的 stdout 事件流（agent_start + message_update + agent_end），
    验证 PiSpecialist 正确解析 + 返 SpecialistRawResult。
    """
    # 构造 pi RPC stdout 事件流
    stdout_lines = [
        _make_event_line({"type": "agent_start"}),
        _make_event_line({
            "type": "message_update",
            "assistantMessageEvent": {"type": "text_delta", "delta": "## What You Did\n"},
        }),
        _make_event_line({
            "type": "message_update",
            "assistantMessageEvent": {"type": "text_delta", "delta": "- Created auth.py\n"},
        }),
        _make_event_line({
            "type": "message_update",
            "assistantMessageEvent": {"type": "text_delta", "delta": "## Success\n- yes, tests pass\n"},
        }),
        _make_event_line({
            "type": "message_update",
            "assistantMessageEvent": {"type": "text_delta", "delta": "## Gaps\n- none\n"},
        }),
        _make_event_line({
            "type": "agent_end",
            "messages": [
                {
                    "role": "assistant",
                    "usage": {
                        "tokens": {
                            "input": 100,
                            "output": 50,
                            "total": 150,
                        }
                    },
                }
            ],
        }),
    ]
    mock_proc = _mock_proc(stdout_lines)

    specialist = PiSpecialist()
    with patch("subprocess.Popen", return_value=mock_proc):
        result = specialist.invoke(_make_request())

    # 验证 SpecialistRawResult
    assert "Created auth.py" in result.raw_output
    assert "tests pass" in result.raw_output
    assert result.usage is not None
    assert result.usage.prompt_tokens == 100
    assert result.usage.total_tokens == 150
    assert result.duration_seconds >= 0


def test_end_to_end_no_builtin_tools():
    """端到端：pi 命令含 --no-builtin-tools（决策 1，禁用 pi 自带工具）。"""
    specialist = PiSpecialist()
    mock_proc = _mock_proc([_make_event_line({"type": "agent_end", "messages": []})])

    with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
        specialist.invoke(_make_request())

    # 验证 Popen 调用含 --no-builtin-tools
    call_args = mock_popen.call_args
    cmd = call_args[0][0] if call_args[0] else call_args[1].get("args", [])
    assert "--no-builtin-tools" in cmd


def test_end_to_end_pi_command_loads_sandbox_bridge():
    """端到端：pi 命令含 -e agent4ml-sandbox-bridge（决策 1，加载 extension）。"""
    specialist = PiSpecialist()
    mock_proc = _mock_proc([_make_event_line({"type": "agent_end", "messages": []})])

    with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
        specialist.invoke(_make_request())

    call_args = mock_popen.call_args
    cmd = call_args[0][0] if call_args[0] else call_args[1].get("args", [])
    assert "-e" in cmd
    e_idx = cmd.index("-e")
    assert "pi-sandbox-bridge" in cmd[e_idx + 1]


def test_end_to_end_pi_env_passes_sandbox_mcp_endpoint():
    """端到端：env 含 AGENT4ML_SANDBOX_MCP_ENDPOINT（决策 1，sandbox_id 绑定）。"""
    specialist = PiSpecialist()
    mock_proc = _mock_proc([_make_event_line({"type": "agent_end", "messages": []})])

    with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant"}):
            specialist.invoke(_make_request(sandbox_id="sb123abc"))

    call_args = mock_popen.call_args
    env = call_args[1].get("env")
    assert env is not None
    assert "AGENT4ML_SANDBOX_MCP_ENDPOINT" in env
    assert "sb123abc" in env["AGENT4ML_SANDBOX_MCP_ENDPOINT"]


def test_pi_specialist_fails_returns_error():
    """pi 崩溃（extension_error 事件）→ 抛 SpecialistCrashError。"""
    specialist = PiSpecialist()
    stdout_lines = [
        _make_event_line({
            "type": "extension_error",
            "error": "extension failed to load",
        }),
    ]
    mock_proc = _mock_proc(stdout_lines)

    with patch("subprocess.Popen", return_value=mock_proc):
        with pytest.raises(SpecialistCrashError, match="extension error"):
            specialist.invoke(_make_request())


def test_pi_specialist_uses_shared_sandbox():
    """决策 1：pi 操作的文件 lead agent 立即可见（物理无隔离）。

    验证：pi 命令的 env 含 AGENT4ML_SANDBOX_MCP_ENDPOINT（指向 SpecialistMcpServer），
    pi extension 通过此 endpoint 调 SpecialistMcpServer 操作 thread sandbox。
    SpecialistMcpServer 在 thread sandbox_id 上下文执行，与 lead agent 共享沙箱。
    """
    specialist = PiSpecialist()
    mock_proc = _mock_proc([_make_event_line({"type": "agent_end", "messages": []})])

    with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant"}):
            specialist.invoke(_make_request(sandbox_id="shared-sb-123"))

    call_args = mock_popen.call_args
    env = call_args[1].get("env")
    assert env is not None
    # SpecialistMcpServer endpoint 含 sandbox_id（共享 thread sandbox）
    endpoint = env["AGENT4ML_SANDBOX_MCP_ENDPOINT"]
    assert "shared-sb-123" in endpoint
    assert "specialist_mcp_server" in endpoint


def test_pi_no_bypass_security_guard():
    """决策 1：pi extension 所有操作走 SpecialistMcpServer（经过 SecurityGuard）。

    验证：pi 命令含 --no-builtin-tools（禁用自带工具）+ -e agent4ml-sandbox-bridge
    （加载 extension 转发到 SpecialistMcpServer）。
    pi 不能用自带 read/write/bash 绕过 SecurityGuard。
    """
    specialist = PiSpecialist()
    mock_proc = _mock_proc([_make_event_line({"type": "agent_end", "messages": []})])

    with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant"}):
            specialist.invoke(_make_request())

    call_args = mock_popen.call_args
    cmd = call_args[0][0] if call_args[0] else call_args[1].get("args", [])
    # 禁用自带工具（不能绕过 SecurityGuard）
    assert "--no-builtin-tools" in cmd
    # 加载 extension（所有操作走 SpecialistMcpServer）
    assert "-e" in cmd
    e_idx = cmd.index("-e")
    assert "pi-sandbox-bridge" in cmd[e_idx + 1]
