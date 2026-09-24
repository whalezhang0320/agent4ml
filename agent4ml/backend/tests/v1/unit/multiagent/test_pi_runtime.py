"""PiRuntime 单测 — RPC 协议 + extension 加载 + env 透传 + prompt 构造（P1）。

验证（tasks.md 1.2）：
- test_invoke_success: mock subprocess，验证 prompt 发送 + events 解析 + final text + usage 提取
- test_build_command: 验证 --no-builtin-tools + -e agent4ml-sandbox-bridge 加载（决策 1）
- test_build_env: 验证凭证 env 透传（国内优先）+ AGENT4ML_SANDBOX_MCP_ENDPOINT（决策 1 + 决策 3）
- test_build_prompt: 验证三段输出格式（决策 5）
- test_pi_not_found: pi 命令不在 PATH，抛 SpecialistStartupError
- test_extension_error: extension_error 事件抛 SpecialistCrashError
- test_timeout: 超时抛 SpecialistTimeoutError
"""
from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from agent4ml.backend.agents.multiagent.exceptions import (
    SpecialistCrashError,
    SpecialistStartupError,
    SpecialistTimeoutError,
)
from agent4ml.backend.agents.multiagent.runtimes.pi_runtime import (
    PiRuntime,
    PiRuntimeConfig,
)
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


def _make_event_line(event_dict: dict) -> str:
    """构造 pi RPC stdout 一行 JSON 事件。"""
    return json.dumps(event_dict)


def _mock_proc(stdout_lines: list[str], returncode: int = 0):
    """构造 mock subprocess.Popen，stdout 返指定 lines。"""
    mock_proc = MagicMock()
    mock_proc.stdout = iter(stdout_lines)
    mock_proc.stderr = iter([])
    mock_proc.stdin = MagicMock()
    mock_proc.wait.return_value = returncode
    mock_proc.terminate.return_value = None
    mock_proc.kill.return_value = None
    return mock_proc


# ---------------------------------------------------------------------------
# _build_command（决策 1：--no-builtin-tools + -e agent4ml-sandbox-bridge）
# ---------------------------------------------------------------------------


def test_build_command_no_builtin_tools():
    """决策 1：命令含 --no-builtin-tools，pi 不能用自带 read/write/edit/bash。"""
    rt = PiRuntime()
    cmd = rt._build_command(_make_request())
    assert "--no-builtin-tools" in cmd


def test_build_command_loads_sandbox_bridge_extension():
    """决策 1：命令含 -e <agent4ml_extension_path>，加载 agent4ml-sandbox-bridge。"""
    rt = PiRuntime()
    cmd = rt._build_command(_make_request())
    assert "-e" in cmd
    # extension 路径含 pi-sandbox-bridge
    e_idx = cmd.index("-e")
    assert e_idx + 1 < len(cmd)
    assert "pi-sandbox-bridge" in cmd[e_idx + 1]
    assert cmd[e_idx + 1].endswith("index.ts")


def test_build_command_with_provider_and_model():
    """config 配 provider + model 时命令含 --provider + --model。"""
    rt = PiRuntime(
        PiRuntimeConfig(provider="deepseek", model="deepseek-chat")
    )
    cmd = rt._build_command(_make_request())
    assert "--provider" in cmd
    assert "deepseek" in cmd
    assert "--model" in cmd
    assert "deepseek-chat" in cmd


def test_build_command_with_thinking_level():
    """config 配 thinking_level 时命令含 --thinking。"""
    rt = PiRuntime(PiRuntimeConfig(thinking_level="high"))
    cmd = rt._build_command(_make_request())
    assert "--thinking" in cmd
    assert "high" in cmd


def test_build_command_no_system_prompt_flag():
    """决策 5：不传 --system-prompt / --append-system-prompt。"""
    rt = PiRuntime()
    cmd = rt._build_command(_make_request())
    assert "--system-prompt" not in cmd
    assert "--append-system-prompt" not in cmd


def test_build_command_default_mode_args():
    """默认 mode_args 含 --mode rpc --no-session。"""
    rt = PiRuntime()
    cmd = rt._build_command(_make_request())
    assert "--mode" in cmd
    assert "rpc" in cmd
    assert "--no-session" in cmd


# ---------------------------------------------------------------------------
# _build_env（决策 1：AGENT4ML_SANDBOX_MCP_ENDPOINT + 决策 3：国内 provider 优先）
# ---------------------------------------------------------------------------


def test_build_env_transmits_deepseek_key(monkeypatch):
    """决策 3：DEEPSEEK_API_KEY 透传（国内 provider 优先）。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    rt = PiRuntime()
    env = rt._build_env(_make_request(sandbox_id="sb123"))
    assert env is not None
    assert env["DEEPSEEK_API_KEY"] == "sk-deepseek"


def test_build_env_transmits_anthropic_key(monkeypatch):
    """ANTHROPIC_API_KEY 透传。"""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    rt = PiRuntime()
    env = rt._build_env(_make_request(sandbox_id="sb123"))
    assert env is not None
    assert env["ANTHROPIC_API_KEY"] == "sk-ant"


def test_build_env_passes_sandbox_mcp_endpoint(monkeypatch):
    """决策 1：request.sandbox_id 非空时 env 含 AGENT4ML_SANDBOX_MCP_ENDPOINT。"""
    for var in ("DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY", "KIMI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")  # 设一个 auth var 触发 merge
    rt = PiRuntime()
    env = rt._build_env(_make_request(sandbox_id="sb123abc"))
    assert env is not None
    assert "AGENT4ML_SANDBOX_MCP_ENDPOINT" in env
    assert "sb123abc" in env["AGENT4ML_SANDBOX_MCP_ENDPOINT"]


def test_build_env_no_sandbox_id_no_endpoint(monkeypatch):
    """request.sandbox_id=None 时 env 不含 AGENT4ML_SANDBOX_MCP_ENDPOINT。"""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    rt = PiRuntime()
    env = rt._build_env(_make_request(sandbox_id=None))
    assert env is not None
    assert "AGENT4ML_SANDBOX_MCP_ENDPOINT" not in env


def test_build_env_no_auth_vars_no_sandbox_returns_none(monkeypatch):
    """无 auth env vars + 无 sandbox_id 时返 None。"""
    for var in (
        "DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY", "KIMI_API_KEY",
        "OPENAI_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY",
        "GROQ_API_KEY", "XAI_API_KEY", "MISTRAL_API_KEY", "TOGETHER_API_KEY",
        "MINIMAX_API_KEY", "XIAOMI_API_KEY", "ZAI_API_KEY",
        "PI_CODING_AGENT_DIR",
    ):
        monkeypatch.delenv(var, raising=False)
    rt = PiRuntime()
    env = rt._build_env(_make_request(sandbox_id=None))
    assert env is None


def test_build_env_merges_parent_env(monkeypatch):
    """_build_env 返 merge 后的 env（父 env + auth vars 覆盖）。"""
    import os
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    parent_path = os.environ.get("PATH", "")
    rt = PiRuntime()
    env = rt._build_env(_make_request(sandbox_id="sb1"))
    assert env is not None
    assert env["PATH"] == parent_path
    assert env["ANTHROPIC_API_KEY"] == "sk-ant"


def test_build_env_passes_pi_coding_agent_dir(monkeypatch):
    """PI_CODING_AGENT_DIR 透传（自定义配置目录）。"""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")  # 触发 merge
    monkeypatch.setenv("PI_CODING_AGENT_DIR", "/custom/pi/dir")
    rt = PiRuntime()
    env = rt._build_env(_make_request(sandbox_id="sb1"))
    assert env is not None
    assert env["PI_CODING_AGENT_DIR"] == "/custom/pi/dir"


# ---------------------------------------------------------------------------
# _build_prompt（决策 5：MVP 不定制 system prompt，靠 user message 塞 context）
# ---------------------------------------------------------------------------


def test_build_prompt_contains_goal():
    """prompt 含 goal。"""
    rt = PiRuntime()
    prompt = rt._build_prompt(_make_request(goal="write auth.py"))
    assert "write auth.py" in prompt


def test_build_prompt_contains_context_summary():
    """prompt 含 context_summary。"""
    rt = PiRuntime()
    prompt = rt._build_prompt(_make_request(context_summary="existing code here"))
    assert "## Context" in prompt
    assert "existing code here" in prompt


def test_build_prompt_contains_success_criteria():
    """prompt 含 success_criteria。"""
    rt = PiRuntime()
    prompt = rt._build_prompt(
        _make_request(success_criteria="tests pass + no breaking changes")
    )
    assert "## Success Criteria" in prompt
    assert "tests pass + no breaking changes" in prompt


def test_build_prompt_contains_three_section_output_format():
    """决策 5：prompt 含三段输出格式（What You Did / Success / Gaps）。"""
    rt = PiRuntime()
    prompt = rt._build_prompt(_make_request())
    assert "## Output Format" in prompt
    assert "## What You Did" in prompt
    assert "## Success" in prompt
    assert "## Gaps" in prompt


# ---------------------------------------------------------------------------
# invoke（RPC 协议 + events 解析 + 异常处理）
# ---------------------------------------------------------------------------


def test_invoke_success_via_mock(monkeypatch):
    """mock subprocess，验证 prompt 发送 + events 解析 + final text + usage 提取。"""
    rt = PiRuntime()
    # 模拟 pi RPC stdout 事件流
    stdout_lines = [
        _make_event_line({"type": "agent_start"}),
        _make_event_line({
            "type": "message_update",
            "assistantMessageEvent": {"type": "text_delta", "delta": "Hello "},
        }),
        _make_event_line({
            "type": "message_update",
            "assistantMessageEvent": {"type": "text_delta", "delta": "world"},
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
    with patch("subprocess.Popen", return_value=mock_proc):
        result = rt.invoke(_make_request())

    assert result.raw_output == "Hello world"
    assert result.usage is not None
    assert result.usage.prompt_tokens == 100
    assert result.usage.completion_tokens == 50
    assert result.usage.total_tokens == 150
    assert result.duration_seconds >= 0


def test_invoke_no_usage_returns_none(monkeypatch):
    """agent_end 事件无 usage 字段时 TokenUsage=None。"""
    rt = PiRuntime()
    stdout_lines = [
        _make_event_line({"type": "agent_start"}),
        _make_event_line({
            "type": "message_update",
            "assistantMessageEvent": {"type": "text_delta", "delta": "ok"},
        }),
        _make_event_line({"type": "agent_end", "messages": []}),
    ]
    mock_proc = _mock_proc(stdout_lines)
    with patch("subprocess.Popen", return_value=mock_proc):
        result = rt.invoke(_make_request())

    assert result.raw_output == "ok"
    assert result.usage is None


def test_invoke_extension_error_raises_crash():
    """extension_error 事件抛 SpecialistCrashError。"""
    rt = PiRuntime()
    stdout_lines = [
        _make_event_line({
            "type": "extension_error",
            "error": "extension failed to load",
        }),
    ]
    mock_proc = _mock_proc(stdout_lines)
    with patch("subprocess.Popen", return_value=mock_proc):
        with pytest.raises(SpecialistCrashError, match="extension error"):
            rt.invoke(_make_request())


def test_invoke_pi_not_found_raises_startup():
    """pi 命令不在 PATH（Popen 抛 FileNotFoundError）→ SpecialistStartupError。"""
    rt = PiRuntime()
    with patch("subprocess.Popen", side_effect=FileNotFoundError("pi not found")):
        with pytest.raises(SpecialistStartupError, match="pi command not found"):
            rt.invoke(_make_request())


def test_invoke_timeout_raises_timeout():
    """pi 子进程超时 → SpecialistTimeoutError。"""
    rt = PiRuntime()

    def _mock_popen(*args, **kwargs):
        mock_proc = MagicMock()
        mock_proc.stdout = iter([])  # 无输出，模拟卡住
        mock_proc.stdin = MagicMock()
        # stdin.flush 会抛 BrokenPipeError，但这里模拟 wait 超时
        mock_proc.stdin.close = MagicMock()
        mock_proc.wait.side_effect = subprocess.TimeoutExpired(cmd="pi", timeout=5)
        mock_proc.terminate.return_value = None
        mock_proc.kill.return_value = None
        # stdout 迭代完无 agent_end，_run_rpc_session 返空，但 invoke 不会超时
        # 这里用 side_effect 让 Popen 抛 TimeoutExpired
        raise subprocess.TimeoutExpired(cmd="pi", timeout=5)

    with patch("subprocess.Popen", side_effect=_mock_popen):
        with pytest.raises(SpecialistTimeoutError):
            rt.invoke(_make_request(timeout_seconds=5))


def test_invoke_skips_invalid_json_lines():
    """stdout 含无效 JSON 行时跳过，不崩。"""
    rt = PiRuntime()
    stdout_lines = [
        "invalid json line\n",
        _make_event_line({"type": "agent_start"}),
        "another invalid\n",
        _make_event_line({
            "type": "message_update",
            "assistantMessageEvent": {"type": "text_delta", "delta": "ok"},
        }),
        _make_event_line({"type": "agent_end", "messages": []}),
    ]
    mock_proc = _mock_proc(stdout_lines)
    with patch("subprocess.Popen", return_value=mock_proc):
        result = rt.invoke(_make_request())

    assert result.raw_output == "ok"


def test_invoke_skips_empty_lines():
    """stdout 含空行时跳过。"""
    rt = PiRuntime()
    stdout_lines = [
        "\n",
        "  \n",
        _make_event_line({
            "type": "message_update",
            "assistantMessageEvent": {"type": "text_delta", "delta": "text"},
        }),
        _make_event_line({"type": "agent_end", "messages": []}),
    ]
    mock_proc = _mock_proc(stdout_lines)
    with patch("subprocess.Popen", return_value=mock_proc):
        result = rt.invoke(_make_request())

    assert result.raw_output == "text"


# ---------------------------------------------------------------------------
# custom config
# ---------------------------------------------------------------------------


def test_custom_command():
    """自定义 command（如 /usr/local/bin/pi）。"""
    rt = PiRuntime(PiRuntimeConfig(command="/usr/local/bin/pi"))
    assert rt._config.command == "/usr/local/bin/pi"


def test_custom_extra_args():
    """自定义 extra_args。"""
    rt = PiRuntime(PiRuntimeConfig(extra_args=("--verbose",)))
    cmd = rt._build_command(_make_request())
    assert "--verbose" in cmd
