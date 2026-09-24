"""L2 bootstrap tests - setup_l2 assembly + daemon thread start + enabled=false.

Design (spec.md multiagent-core setup_multiagent Requirement + 42 doc S13):
- config.l2.enabled=true: all L2 components assembled + daemon thread started
- config.l2.enabled=false: l2_setup=None, L1 behavior unchanged
- L2 components injected into L1 make_specialist_tool + OrchestrationMiddleware
"""
from __future__ import annotations

import pytest

from agent4ml.backend.agents.multiagent.config import MultiAgentConfig
from agent4ml.backend.agents.multiagent.evolution.bootstrap import setup_l2
from agent4ml.backend.agents.multiagent.metrics import MultiAgentMetricsStore


@pytest.fixture
def metrics_store(tmp_path):
    return MultiAgentMetricsStore(db_path=str(tmp_path / "test_l2_bootstrap.db"))


# -- config.l2.enabled=false -> None -----------------------------------------


def test_setup_l2_disabled_returns_none(metrics_store):
    """config.l2.enabled=false -> setup_l2 returns None."""
    result = setup_l2(MultiAgentConfig(), metrics_store)
    assert result is None


# -- config.l2.enabled=true -> all components -------------------------------


def test_setup_l2_enabled_assembles_all_components(metrics_store):
    """config.l2.enabled=true -> all L2 components constructed."""
    config = MultiAgentConfig()
    # Override l2.enabled via direct construction (frozen dataclass)
    from dataclasses import replace
    config = replace(config, l2=replace(config.l2, enabled=True))

    result = setup_l2(config, metrics_store)
    assert result is not None
    assert result.version_dag is not None
    assert result.promotion_gate is not None
    assert result.evolution_mutator is not None
    assert result.trigger_manager is not None
    assert result.l2_trigger_middleware is not None
    assert result.budget_guard is not None
    assert result.metrics_l2 is not None
    assert result.worker is not None
    assert result.task_queue is not None


def test_setup_l2_enabled_injects_correct_config(metrics_store):
    """L2 components use config values (cron_interval, cooldown, thresholds)."""
    config = MultiAgentConfig()
    from dataclasses import replace
    config = replace(config, l2=replace(
        config.l2, enabled=True,
        cron_interval_hours=12.0, cooldown_seconds=7200.0,
        failure_threshold=10,
    ))

    result = setup_l2(config, metrics_store)
    assert result is not None
    # TriggerManager config
    assert result.trigger_manager._cooldown_seconds == 7200.0
    assert result.trigger_manager._cron_interval_seconds == 12.0 * 3600
    assert result.trigger_manager._thresholds.failure_threshold == 10


def test_setup_l2_budget_guard_uses_budget_config(metrics_store):
    """BudgetGuard uses per-specialist limits from config.budget."""
    config = MultiAgentConfig()
    from dataclasses import replace
    config = replace(config, l2=replace(config.l2, enabled=True))

    result = setup_l2(config, metrics_store)
    assert result is not None
    # BudgetGuard should have limits for codex/claude/subagent/pi
    assert "codex" in result.budget_guard._limits
    assert result.budget_guard._limits["codex"].per_day_tokens == 200000


def test_setup_l2_worker_not_started_by_setup(metrics_store):
    """setup_l2 constructs worker but does NOT start daemon thread.

    Daemon thread start is responsibility of setup_multiagent (L1 bootstrap),
    not setup_l2 (to keep setup_l2 pure assembly + testable).
    """
    config = MultiAgentConfig()
    from dataclasses import replace
    config = replace(config, l2=replace(config.l2, enabled=True))

    result = setup_l2(config, metrics_store)
    assert result is not None
    assert not result.worker.is_running  # not started yet


def test_setup_l2_worker_start_stop(metrics_store):
    """Worker can be started + stopped after setup_l2."""
    config = MultiAgentConfig()
    from dataclasses import replace
    config = replace(config, l2=replace(config.l2, enabled=True))

    result = setup_l2(config, metrics_store)
    result.worker.start()
    assert result.worker.is_running
    result.worker.stop()
    assert not result.worker.is_running


# -- setup_multiagent integration (L2 enabled=true) -------------------------


def test_setup_multiagent_l2_enabled_starts_daemon(tmp_path):
    """setup_multiagent with l2.enabled=true."""
    from agent4ml.backend.agents.multiagent.bootstrap import setup_multiagent
    from agent4ml.backend.agents.multiagent.config import MultiAgentConfig
    from dataclasses import replace

    config = replace(
        MultiAgentConfig(),
        enabled=True,
        specialists_use=("subagent",),  # only subagent (zero-config)
        l2=replace(MultiAgentConfig().l2, enabled=True),
        metrics_db_path=str(tmp_path / "test_setup.db"),
    )

    setup = setup_multiagent(config, agent_factory=lambda: None)
    assert setup.l2_setup is not None
    assert setup.l2_setup.worker.is_running
    # Cleanup
    setup.l2_setup.worker.stop()


def test_setup_multiagent_l2_disabled_no_daemon(tmp_path):
    """setup_multiagent with l2.enabled=false -> no daemon, L1 behavior unchanged."""
    from agent4ml.backend.agents.multiagent.bootstrap import setup_multiagent
    from agent4ml.backend.agents.multiagent.config import MultiAgentConfig

    config = MultiAgentConfig(
        enabled=True,
        specialists_use=("subagent",),
        metrics_db_path=str(tmp_path / "test_setup.db"),
    )

    setup = setup_multiagent(config, agent_factory=lambda: None)
    assert setup.l2_setup is None
