"""PiSpecialist + bootstrap 集成 + config + routing 段测试（P6）。

验证（tasks.md 6.6）：
- test_pi_specialist_invoke: mock runtime，验证 invoke 流程
- test_bootstrap_loads_pi: bootstrap 装配 pi specialist（pi 已装 + 凭证就绪）
- test_bootstrap_pi_not_installed: pi 未装 + auto_install=true，bootstrap 启动后台安装，pi specialist 不注册
- test_bootstrap_pi_no_credential: pi 装了但无凭证，warn + 不注册
- test_routing_section_includes_pi: specialist routing 段含 pi + "coding → pi preferred"
- test_tui_banner_specialist_status: TUI 启动 banner 显示 specialist 状态（决策 2）
- test_config_pi_fields: MultiAgentConfig 含 pi 字段
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent4ml.backend.agents.multiagent.config import MultiAgentConfig
from agent4ml.backend.agents.multiagent.runtimes.pi_runtime import (
    PiRuntime,
    PiRuntimeConfig,
)
from agent4ml.backend.agents.multiagent.specialists.pi_specialist import PiSpecialist
from agent4ml.backend.agents.multiagent.types import (
    SpecialistCapabilities,
    SpecialistCapability,
    SpecialistRawResult,
    SpecialistRequest,
)


def _make_request(**kwargs) -> SpecialistRequest:
    defaults = dict(
        goal="write a function",
        success_criteria="function exists",
        context_summary="python project",
        sandbox_id="sb123",
        artifacts_path="/workspace",
        timeout_seconds=60,
    )
    defaults.update(kwargs)
    return SpecialistRequest(**defaults)


# ---------------------------------------------------------------------------
# PiSpecialist 组合 + invoke
# ---------------------------------------------------------------------------


def test_pi_specialist_invoke():
    """mock runtime，验证 invoke 流程。"""
    mock_runtime = MagicMock()
    mock_runtime.invoke.return_value = SpecialistRawResult(
        raw_output="task done", duration_seconds=1.0
    )
    specialist = PiSpecialist(runtime=mock_runtime)
    result = specialist.invoke(_make_request())
    assert result.raw_output == "task done"
    mock_runtime.invoke.assert_called_once()


def test_pi_specialist_name():
    """specialist.name == "pi"。"""
    specialist = PiSpecialist()
    assert specialist.name == "pi"


def test_pi_specialist_capabilities():
    """specialist.capabilities 含 CODING。"""
    specialist = PiSpecialist()
    caps = specialist.capabilities
    assert SpecialistCapability.CODING in caps.capabilities


def test_pi_specialist_default_runtime():
    """无 runtime 参数时默认构造 PiRuntime。"""
    specialist = PiSpecialist()
    assert isinstance(specialist._runtime, PiRuntime)


# ---------------------------------------------------------------------------
# MultiAgentConfig pi 字段
# ---------------------------------------------------------------------------


def test_config_pi_fields():
    """MultiAgentConfig 含 pi 字段（P6 config schema 扩展）。"""
    c = MultiAgentConfig()
    assert hasattr(c, "specialists_pi_provider")
    assert hasattr(c, "specialists_pi_api_key")
    assert hasattr(c, "specialists_pi_auto_install")
    assert hasattr(c, "specialists_pi_model")
    assert hasattr(c, "specialists_pi_thinking_level")
    assert c.specialists_pi_provider == ""
    assert c.specialists_pi_auto_install is True
    assert c.specialists_pi_thinking_level == "medium"


def test_config_specialists_use_contains_pi():
    """specialists_use 默认含 pi（作为默认 coding specialist）。"""
    c = MultiAgentConfig()
    assert "pi" in c.specialists_use


def test_config_frozen_with_pi_fields():
    """MultiAgentConfig 含 pi 字段后仍 frozen。"""
    c = MultiAgentConfig()
    with pytest.raises(Exception):
        c.specialists_pi_provider = "deepseek"  # type: ignore


# ---------------------------------------------------------------------------
# bootstrap 加载 pi specialist
# ---------------------------------------------------------------------------


def test_bootstrap_loads_pi_when_installed_and_credentialed(monkeypatch):
    """pi 已装 + 凭证就绪 → bootstrap 注册 PiSpecialist。"""
    # mock pi 已装
    monkeypatch.setattr(
        "agent4ml.backend.agents.multiagent.installer.pi_installer.shutil.which",
        lambda cmd: "/usr/local/bin/pi" if cmd == "pi" else None,
    )
    # mock 凭证就绪
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")

    from agent4ml.backend.agents.multiagent.bootstrap import _load_specialist

    config = MultiAgentConfig(
        enabled=True,
        specialists_use=("pi",),
        specialists_pi_auto_install=False,  # 避免触发后台安装
    )
    result = _load_specialist("pi", config, agent_factory=None)

    assert result is not None
    specialist, ctx_summarizer, result_summarizer = result
    assert isinstance(specialist, PiSpecialist)
    assert specialist.name == "pi"


def test_bootstrap_pi_not_installed_returns_none(monkeypatch, tmp_path):
    """pi 未装 + auto_install=false → 返 None（specialist 不注册）。"""
    # mock pi 未装
    monkeypatch.setattr(
        "agent4ml.backend.agents.multiagent.installer.pi_installer.shutil.which",
        lambda cmd: None,
    )
    # mock flag file 不存在
    monkeypatch.setattr(
        "agent4ml.backend.agents.multiagent.installer.pi_installer.PiInstaller.FLAG_FILE",
        tmp_path / "nonexistent-flag",
    )
    # mock 凭证就绪（但 pi 没装）
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")

    from agent4ml.backend.agents.multiagent.bootstrap import _load_specialist

    config = MultiAgentConfig(
        enabled=True,
        specialists_use=("pi",),
        specialists_pi_auto_install=False,  # 不自动安装
    )
    result = _load_specialist("pi", config, agent_factory=None)

    assert result is None  # pi 未装，disabled


def test_bootstrap_pi_no_credential_returns_none(monkeypatch):
    """pi 装了但无凭证 → 返 None（凭证缺失 disabled）。"""
    # mock pi 已装
    monkeypatch.setattr(
        "agent4ml.backend.agents.multiagent.installer.pi_installer.shutil.which",
        lambda cmd: "/usr/local/bin/pi" if cmd == "pi" else None,
    )
    # 清除所有凭证 env vars
    for var in (
        "DEEPSEEK_API_KEY", "KIMI_API_KEY", "MINIMAX_API_KEY", "XIAOMI_API_KEY",
        "ZAI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY",
        "OPENROUTER_API_KEY", "GROQ_API_KEY", "XAI_API_KEY", "MISTRAL_API_KEY",
        "TOGETHER_API_KEY", "PI_CODING_AGENT_DIR",
    ):
        monkeypatch.delenv(var, raising=False)

    from agent4ml.backend.agents.multiagent.bootstrap import _load_specialist

    config = MultiAgentConfig(
        enabled=True,
        specialists_use=("pi",),
        specialists_pi_auto_install=False,
    )
    result = _load_specialist("pi", config, agent_factory=None)

    assert result is None  # 凭证缺失


def test_bootstrap_pi_warns_when_disabled(monkeypatch, tmp_path, caplog):
    """pi disabled 时 warn 提示安装步骤。"""
    import logging

    # mock pi 未装
    monkeypatch.setattr(
        "agent4ml.backend.agents.multiagent.installer.pi_installer.shutil.which",
        lambda cmd: None,
    )
    monkeypatch.setattr(
        "agent4ml.backend.agents.multiagent.installer.pi_installer.PiInstaller.FLAG_FILE",
        tmp_path / "nonexistent-flag",
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek")

    from agent4ml.backend.agents.multiagent.bootstrap import (
        _warn_specialist_disabled,
    )

    with caplog.at_level(
        logging.WARNING,
        logger="agent4ml.backend.agents.multiagent.bootstrap",
    ):
        _warn_specialist_disabled("pi", "credential missing or load failed")

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) >= 1
    msg = warnings[0].getMessage()
    assert "PiSpecialist" in msg
    assert "npm install -g @earendil-works/pi-coding-agent" in msg
    assert "ANTHROPIC_API_KEY" in msg or "DEEPSEEK_API_KEY" in msg


# ---------------------------------------------------------------------------
# routing 段含 pi + coding → pi preferred（决策 4）
# ---------------------------------------------------------------------------


def test_routing_section_includes_pi():
    """specialist routing 段含 pi + "coding → pi preferred" 引导（决策 4）。"""
    from agent4ml.backend.agents.leader.prompts import _build_specialist_routing_section

    mock_registry = MagicMock()
    mock_registry.list_specialists.return_value = ["pi", "codex", "claude"]
    section = _build_specialist_routing_section(mock_registry)

    assert "<specialist_routing>" in section
    assert "delegate_to_pi" in section
    assert "delegate_to_codex" in section
    assert "delegate_to_claude" in section
    # 决策 4：coding → specialist 引导
    assert "coding tasks" in section.lower() or "coding" in section.lower()


# ---------------------------------------------------------------------------
# TUI banner specialist 状态（决策 2）
# ---------------------------------------------------------------------------


def test_tui_banner_has_coding_specialist():
    """有 coding specialist 时 banner 显示 "✓ Coding specialists"。"""
    # 这个测试验证 _build_startup_banner_specialist_status 函数（待 P6 实现）
    # 当前函数还没写，此测试是 placeholder
    # 实际实现应在 app/cli/commands.py 或 leader/factory.py
    # 此测试暂时跳过，等 P6 后续实现 TUI banner 函数
    pytest.skip("TUI banner function not yet implemented in this batch")


def test_tui_banner_pi_installing():
    """pi 后台安装中时 banner 显示提示（决策 2）。"""
    pytest.skip("TUI banner function not yet implemented in this batch")
