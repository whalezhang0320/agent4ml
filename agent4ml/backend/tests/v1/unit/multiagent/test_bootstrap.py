"""Multi-Agent bootstrap 单测 — 反射加载 + 凭证缺失 disabled + enabled=false 不装配。"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent4ml.backend.agents.multiagent.bootstrap import (
    MultiAgentSetup,
    _EMPTY_SETUP,
    setup_multiagent,
)
from agent4ml.backend.agents.multiagent.config import (
    MultiAgentConfig,
    STARTUP_ONLY_FIELDS,
    load_multiagent_config,
)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


def test_config_defaults():
    c = MultiAgentConfig()
    assert c.enabled is True
    assert "pi" in c.specialists_use  # P6: pi 作为默认 coding specialist
    assert "codex" in c.specialists_use
    assert c.max_steps == 50
    assert c.subagent_max_steps == 20
    assert c.specialists_pi_auto_install is True  # P6: 默认自动安装


def test_config_frozen():
    c = MultiAgentConfig()
    with pytest.raises(Exception):
        c.enabled = True


def test_startup_only_fields():
    assert "enabled" in STARTUP_ONLY_FIELDS
    assert "specialists_use" in STARTUP_ONLY_FIELDS
    assert "metrics_db_path" in STARTUP_ONLY_FIELDS


def test_load_config_enabled_by_default(monkeypatch):
    monkeypatch.delenv("AGENT4ML_MULTIAGENT_ENABLED", raising=False)
    c = load_multiagent_config()
    assert c.enabled is True


def test_load_config_enabled(monkeypatch):
    monkeypatch.setenv("AGENT4ML_MULTIAGENT_ENABLED", "true")
    c = load_multiagent_config()
    assert c.enabled is True


# ---------------------------------------------------------------------------
# setup_multiagent — enabled=false
# ---------------------------------------------------------------------------


def test_setup_disabled_returns_empty():
    config = MultiAgentConfig(enabled=False)
    setup = setup_multiagent(config)
    assert setup.specialist_registry is None
    assert setup.subagent_provider is None
    assert setup.metrics_store is None
    assert setup.orchestration_middleware is None
    assert setup.specialist_tools == ()


def test_setup_disabled_is_empty_setup():
    config = MultiAgentConfig(enabled=False)
    setup = setup_multiagent(config)
    assert setup is _EMPTY_SETUP


# ---------------------------------------------------------------------------
# setup_multiagent — enabled=true
# ---------------------------------------------------------------------------


def test_setup_enabled_creates_metrics_and_middleware(tmp_path):
    config = MultiAgentConfig(
        enabled=True,
        specialists_use=(),
        metrics_db_path=str(tmp_path / "test.db"),
    )
    setup = setup_multiagent(config)
    assert setup.metrics_store is not None
    assert setup.orchestration_middleware is not None
    assert setup.specialist_registry is not None


def test_setup_enabled_no_specialists_no_tools(tmp_path):
    config = MultiAgentConfig(
        enabled=True,
        specialists_use=(),
        metrics_db_path=str(tmp_path / "test.db"),
    )
    setup = setup_multiagent(config)
    assert setup.specialist_tools == ()


def test_setup_credential_missing_disables_specialist(tmp_path, monkeypatch):
    """codex/claude credential missing → specialist disabled."""
    monkeypatch.delenv("CODEX_AUTH_PATH", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_CREDENTIALS_PATH", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    config = MultiAgentConfig(
        enabled=True,
        specialists_use=("codex", "claude"),
        metrics_db_path=str(tmp_path / "test.db"),
    )
    setup = setup_multiagent(config)
    # No tools generated (credentials missing)
    assert len(setup.specialist_tools) == 0
    # Registry empty (no specialists registered)
    assert len(setup.specialist_registry) == 0


def test_setup_subagent_no_credential_needed(tmp_path):
    """subagent doesn't need credentials."""
    config = MultiAgentConfig(
        enabled=True,
        specialists_use=("subagent",),
        metrics_db_path=str(tmp_path / "test.db"),
    )
    setup = setup_multiagent(config)
    assert len(setup.specialist_tools) == 1
    assert setup.specialist_tools[0].name == "delegate_to_subagent"
    assert len(setup.specialist_registry) == 1


def test_setup_subagent_with_agent_factory(tmp_path):
    mock_factory = MagicMock(return_value=MagicMock())
    config = MultiAgentConfig(
        enabled=True,
        specialists_use=("subagent",),
        metrics_db_path=str(tmp_path / "test.db"),
    )
    setup = setup_multiagent(config, agent_factory=mock_factory)
    # subagent specialist → one delegate_to_subagent tool (not two)
    assert len(setup.specialist_tools) == 1
    assert setup.specialist_tools[0].name == "delegate_to_subagent"
    # subagent_provider not set separately (specialist handles it)
    assert setup.subagent_provider is None


def test_setup_unknown_specialist_skipped(tmp_path):
    config = MultiAgentConfig(
        enabled=True,
        specialists_use=("unknown_specialist",),
        metrics_db_path=str(tmp_path / "test.db"),
    )
    setup = setup_multiagent(config)
    assert len(setup.specialist_tools) == 0
    assert len(setup.specialist_registry) == 0


def test_setup_multiagent_setup_frozen():
    with pytest.raises(Exception):
        _EMPTY_SETUP.specialist_tools = []
