"""L1 modification tests - MetricsView Protocol + failure_category + config l2/budget + tools budget_guard + middleware L2 slots.

Design (spec.md multiagent-core MODIFIED Requirements + Batch 12):
- MultiAgentMetricsStore implements MetricsView Protocol (isinstance check)
- BaseResultSummarizer outputs failure_category (heuristic classification)
- ContextSummarizer.summarize accepts template param (backward compat)
- make_specialist_tool accepts version_dag + budget_guard (backward compat)
- OrchestrationMiddleware accepts L2 slots (default None)
- MultiAgentConfig has l2 + budget section (default enabled=false)
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent4ml.backend.agents.multiagent.config import (
    BudgetConfig,
    L2Config,
    MultiAgentConfig,
    SpecialistBudgetLimit,
)
from agent4ml.backend.agents.multiagent.evolution.metrics_view import MetricsView
from agent4ml.backend.agents.multiagent.metrics import MultiAgentMetricsStore
from agent4ml.backend.agents.multiagent.middleware import OrchestrationMiddleware
from agent4ml.backend.agents.multiagent.summarizers.result.base import BaseResultSummarizer
from agent4ml.backend.agents.multiagent.types import SpecialistResult


# -- MetricsView Protocol implementation ---------------------------------------


def test_metrics_store_implements_metrics_view(tmp_path):
    """MultiAgentMetricsStore implements MetricsView Protocol (isinstance True)."""
    store = MultiAgentMetricsStore(db_path=str(tmp_path / "test.db"))
    assert isinstance(store, MetricsView)


def test_get_specialist_metrics_returns_dict(tmp_path):
    """get_specialist_metrics returns dict snapshot or None."""
    store = MultiAgentMetricsStore(db_path=str(tmp_path / "test.db"))
    store.record_selection("codex")
    store.record_invoked("codex")
    store.record_completion("codex")
    snap = store.get_specialist_metrics("codex")
    assert snap is not None
    assert isinstance(snap, dict)
    assert snap["specialist_name"] == "codex"
    assert snap["total_invoked"] == 1
    assert snap["completion_rate"] == 1.0


def test_get_specialist_metrics_none_for_unknown(tmp_path):
    """get_specialist_metrics returns None for unknown specialist."""
    store = MultiAgentMetricsStore(db_path=str(tmp_path / "test.db"))
    assert store.get_specialist_metrics("unknown") is None


def test_get_global_metrics_returns_dict(tmp_path):
    """get_global_metrics returns dict with totals."""
    store = MultiAgentMetricsStore(db_path=str(tmp_path / "test.db"))
    store.record_selection("codex")
    store.record_invoked("codex")
    snap = store.get_global_metrics()
    assert isinstance(snap, dict)
    assert snap["total_calls"] == 1
    assert snap["total_selections"] == 1


def test_get_failure_categories_empty(tmp_path):
    """get_failure_categories returns empty dict when no failures recorded."""
    store = MultiAgentMetricsStore(db_path=str(tmp_path / "test.db"))
    cats = store.get_failure_categories()
    assert cats == {}


def test_get_failure_categories_with_data(tmp_path):
    """get_failure_categories returns counts by category."""
    store = MultiAgentMetricsStore(db_path=str(tmp_path / "test.db"))
    store.record_judgment(
        "run1", "codex", success=False,
        failure_category="context_insufficient",
    )
    store.record_judgment(
        "run2", "codex", success=False,
        failure_category="context_insufficient",
    )
    cats = store.get_failure_categories()
    from agent4ml.backend.agents.multiagent.evolution.types import FailureCategory
    assert cats.get(FailureCategory.CONTEXT_INSUFFICIENT) == 2


def test_list_specialists(tmp_path):
    """list_specialists returns names with records."""
    store = MultiAgentMetricsStore(db_path=str(tmp_path / "test.db"))
    store.record_selection("codex")
    store.record_selection("claude")
    names = store.list_specialists()
    assert "codex" in names
    assert "claude" in names


# -- BaseResultSummarizer failure_category -----------------------------------


def test_result_summarizer_success_returns_none_category():
    """success=True -> failure_category=None."""
    summarizer = BaseResultSummarizer(specialist_name="codex")
    result = summarizer.summarize(
        raw_output="task completed",
        artifacts=[__import__("agent4ml.backend.agents.multiagent.types", fromlist=["ArtifactRef"]).ArtifactRef(
            path="/p", artifact_type="code", specialist_name="codex"
        )],
        goal="g",
        success_criteria="sc",
    )
    assert result.success is True
    assert result.failure_category is None


def test_result_summarizer_failure_context_keyword():
    """success=False + raw_output contains 'context' -> context_insufficient."""
    summarizer = BaseResultSummarizer(specialist_name="codex")
    result = summarizer.summarize(
        raw_output="missing context for the task",
        artifacts=[],
        goal="g",
        success_criteria="sc",
    )
    assert result.success is False
    assert result.failure_category == "context_insufficient"


def test_result_summarizer_failure_ability_keyword():
    """success=False + 'skill' -> ability_insufficient."""
    summarizer = BaseResultSummarizer(specialist_name="codex")
    result = summarizer.summarize(
        raw_output="lack of skill to complete",
        artifacts=[],
        goal="g",
        success_criteria="sc",
    )
    assert result.failure_category == "ability_insufficient"


def test_result_summarizer_failure_goal_keyword():
    """success=False + 'goal' -> goal_unclear."""
    summarizer = BaseResultSummarizer(specialist_name="codex")
    result = summarizer.summarize(
        raw_output="goal is unclear",
        artifacts=[],
        goal="g",
        success_criteria="sc",
    )
    assert result.failure_category == "goal_unclear"


def test_result_summarizer_failure_sandbox_keyword():
    """success=False + 'sandbox' -> sandbox_issue."""
    summarizer = BaseResultSummarizer(specialist_name="codex")
    result = summarizer.summarize(
        raw_output="sandbox timeout occurred",
        artifacts=[],
        goal="g",
        success_criteria="sc",
    )
    assert result.failure_category == "sandbox_issue"


def test_result_summarizer_failure_default_ability():
    """success=False + no keyword -> ability_insufficient (default)."""
    summarizer = BaseResultSummarizer(specialist_name="codex")
    result = summarizer.summarize(
        raw_output="something went wrong",
        artifacts=[],
        goal="g",
        success_criteria="sc",
    )
    assert result.failure_category == "ability_insufficient"


# -- config l2 + budget section ----------------------------------------------


def test_config_l2_default_disabled():
    """MultiAgentConfig() default l2.enabled=false."""
    config = MultiAgentConfig()
    assert config.l2.enabled is False
    assert config.l2.cron_interval_hours == 6.0
    assert config.l2.cooldown_seconds == 3600.0
    assert config.l2.anti_loop_window == 5
    assert config.l2.failure_threshold == 5
    assert config.l2.eval_timeout_seconds == 1800.0


def test_config_budget_defaults():
    """MultiAgentConfig() default budget limits."""
    config = MultiAgentConfig()
    assert config.budget.codex.per_day_tokens == 200000
    assert config.budget.codex.per_day_cost_usd == 20.0
    assert config.budget.codex.per_day_calls == 50
    assert config.budget.warning_threshold == 0.8


def test_config_startup_only_fields_includes_l2_enabled():
    """STARTUP_ONLY_FIELDS includes l2.enabled."""
    from agent4ml.backend.agents.multiagent.config import STARTUP_ONLY_FIELDS
    assert "l2.enabled" in STARTUP_ONLY_FIELDS
    assert "enabled" in STARTUP_ONLY_FIELDS  # existing L1 field preserved


def test_l2_config_defaults():
    """L2Config defaults (R4)."""
    c = L2Config()
    assert c.enabled is False
    assert c.cron_interval_hours == 6.0
    assert c.degradation_threshold == 0.4
    assert c.cost_alert_usd == 1.0
    assert c.latency_alert_seconds == 300.0


def test_budget_config_defaults():
    """BudgetConfig defaults."""
    b = BudgetConfig()
    assert b.codex.per_day_tokens == 200000
    assert b.warning_threshold == 0.8


# -- OrchestrationMiddleware L2 slots -----------------------------------------


def test_middleware_l2_slots_default_none():
    """OrchestrationMiddleware L2 slots default None."""
    mw = OrchestrationMiddleware()
    assert mw._l2_trigger_middleware is None
    assert mw._budget_guard is None


def test_middleware_l2_slots_accept_injection():
    """OrchestrationMiddleware accepts L2 slots."""
    fake_trigger = SimpleNamespace(name="l2_trigger")
    fake_budget = SimpleNamespace(name="budget_guard")
    mw = OrchestrationMiddleware(
        l2_trigger_middleware=fake_trigger,
        budget_guard=fake_budget,
    )
    assert mw._l2_trigger_middleware is fake_trigger
    assert mw._budget_guard is fake_budget


# -- make_specialist_tool version_dag + budget_guard (backward compat) -------


def test_make_specialist_tool_accepts_version_dag_budget_guard():
    """make_specialist_tool accepts version_dag + budget_guard (default None)."""
    from unittest.mock import MagicMock
    from agent4ml.backend.agents.multiagent.tools import make_specialist_tool
    specialist = MagicMock()
    ctx_summarizer = MagicMock()
    ctx_summarizer.summarize.return_value = "context"
    result_summarizer = MagicMock()
    result_summarizer.summarize.return_value = SpecialistResult(
        specialist_name="codex", summary="ok", success=True
    )
    tool = make_specialist_tool(
        "codex", specialist, ctx_summarizer, result_summarizer,
        version_dag=None, budget_guard=None,
    )
    assert tool is not None


def test_make_specialist_tool_budget_guard_exceeded():
    """budget_guard 超限 -> tool returns BudgetExceeded JSON."""
    from unittest.mock import MagicMock
    from agent4ml.backend.agents.multiagent.tools import (
        _current_state,
        make_specialist_tool,
        set_current_state,
    )
    specialist = MagicMock()
    ctx_summarizer = MagicMock()
    ctx_summarizer.summarize.return_value = "context"
    result_summarizer = MagicMock()
    fake_budget = MagicMock()
    fake_budget.check_and_record.return_value = SimpleNamespace(
        allowed=False,
        reason="daily_cost_exceeded",
        remaining=SimpleNamespace(tokens=0, cost_usd=0.0, calls=0),
        fallback_target="lead",
    )
    tool = make_specialist_tool(
        "codex", specialist, ctx_summarizer, result_summarizer,
        budget_guard=fake_budget,
    )
    # Call the tool (it's a BaseTool, invoke)
    result = tool.invoke({"goal": "g", "success_criteria": "sc"})
    assert isinstance(result, str)
    data = json.loads(result)
    assert data["success"] is False
    assert data["error"]["type"] == "BudgetExceeded"
    assert data["error"]["fallback_target"] == "lead"
    # specialist not called (budget exceeded)
    specialist.invoke.assert_not_called()


# -- ContextSummarizer template param (backward compat) ----------------------


def test_context_summarizer_accepts_template_none():
    """ContextSummarizer.summarize accepts template=None (backward compat)."""
    from agent4ml.backend.agents.multiagent.summarizers.context.codex_context_summarizer import (
        CodexContextSummarizer,
    )
    summarizer = CodexContextSummarizer()
    state = {"messages": []}
    result = summarizer.summarize(state, "goal", "criteria", template=None)
    assert isinstance(result, str)
    assert len(result) > 0
