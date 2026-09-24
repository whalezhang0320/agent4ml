"""OrchestrationMetricsL2 unit tests - 11 event types + payload_json + not ThreadState + query.

Design (spec.md OrchestrationMetricsL2 Requirement + 42 doc S7.11 + R7.1):
- 11 event types: trigger / evolution_start / mutator_call / evolution_failed /
  promotion_decision / eval_executed / eval_failed / version_rollback /
  blocked_marked / budget_warning / budget_exceeded
- Write to multiagent.db l2_metrics table (not ThreadState, INV-38)
- No active push, user inspects via CLI (INV-39)
"""
from __future__ import annotations

from pathlib import Path

import pytest

from agent4ml.backend.agents.multiagent.evolution.metrics_l2 import (
    EVENT_BLOCKED_MARKED,
    EVENT_BUDGET_EXCEEDED,
    EVENT_BUDGET_WARNING,
    EVENT_EVAL_EXECUTED,
    EVENT_EVAL_FAILED,
    EVENT_EVOLUTION_FAILED,
    EVENT_EVOLUTION_START,
    EVENT_MUTATOR_CALL,
    EVENT_PROMOTION_DECISION,
    EVENT_TRIGGER,
    EVENT_VERSION_ROLLBACK,
    OrchestrationMetricsL2,
)


@pytest.fixture
def metrics(tmp_path):
    db_path = str(tmp_path / "test_l2_metrics.db")
    return OrchestrationMetricsL2(db_path=db_path)


# -- 11 event types -----------------------------------------------------------


def test_event_types_count(metrics):
    assert len(metrics.event_types) == 11


def test_event_types_values(metrics):
    types = set(metrics.event_types)
    assert EVENT_TRIGGER in types
    assert EVENT_EVOLUTION_START in types
    assert EVENT_MUTATOR_CALL in types
    assert EVENT_EVOLUTION_FAILED in types
    assert EVENT_PROMOTION_DECISION in types
    assert EVENT_EVAL_EXECUTED in types
    assert EVENT_EVAL_FAILED in types
    assert EVENT_VERSION_ROLLBACK in types
    assert EVENT_BLOCKED_MARKED in types
    assert EVENT_BUDGET_WARNING in types
    assert EVENT_BUDGET_EXCEEDED in types


# -- record_trigger -----------------------------------------------------------


def test_record_trigger(metrics):
    metrics.record_trigger("periodic", "6h cron", "default")
    rows = metrics.query_by_event_type(EVENT_TRIGGER)
    assert len(rows) == 1
    assert rows[0]["payload"]["trigger_source"] == "periodic"
    assert rows[0]["payload"]["trigger_detail"] == "6h cron"
    assert rows[0]["payload"]["profile"] == "default"


# -- record_evolution_start ---------------------------------------------------


def test_record_evolution_start(metrics):
    metrics.record_evolution_start("exp_001", "context_summary", "v1")
    rows = metrics.query_by_event_type(EVENT_EVOLUTION_START)
    assert len(rows) == 1
    assert rows[0]["payload"]["experiment_id"] == "exp_001"
    assert rows[0]["payload"]["artifact_type"] == "context_summary"
    assert rows[0]["payload"]["from_version"] == "v1"


# -- record_mutator_call ------------------------------------------------------


def test_record_mutator_call(metrics):
    metrics.record_mutator_call(
        llm_model="gpt-4",
        input_tokens=1000,
        output_tokens=500,
        latency_ms=2500.0,
        retries=1,
    )
    rows = metrics.query_by_event_type(EVENT_MUTATOR_CALL)
    assert len(rows) == 1
    assert rows[0]["payload"]["llm_model"] == "gpt-4"
    assert rows[0]["payload"]["input_tokens"] == 1000
    assert rows[0]["payload"]["output_tokens"] == 500
    assert rows[0]["payload"]["latency_ms"] == 2500.0
    assert rows[0]["payload"]["retries"] == 1


# -- record_evolution_failed --------------------------------------------------


def test_record_evolution_failed(metrics):
    metrics.record_evolution_failed("json_parse", "exp_001", "LLM returned invalid JSON")
    rows = metrics.query_by_event_type(EVENT_EVOLUTION_FAILED)
    assert len(rows) == 1
    assert rows[0]["payload"]["failure_type"] == "json_parse"
    assert rows[0]["payload"]["experiment_id"] == "exp_001"


# -- record_promotion_decision ------------------------------------------------


def test_record_promotion_decision(metrics):
    metrics.record_promotion_decision("accept", 0.8, 0.4, 0.6, 0.88)
    rows = metrics.query_by_event_type(EVENT_PROMOTION_DECISION)
    assert len(rows) == 1
    assert rows[0]["payload"]["decision"] == "accept"
    assert rows[0]["payload"]["candidate_score"] == 0.8
    assert rows[0]["payload"]["baseline_score"] == 0.4
    assert rows[0]["payload"]["ci_low"] == 0.6
    assert rows[0]["payload"]["ci_high"] == 0.88


# -- record_eval_executed -----------------------------------------------------


def test_record_eval_executed(metrics):
    metrics.record_eval_executed("longitudinal_pairs", 15, 30000.0, 0)
    rows = metrics.query_by_event_type(EVENT_EVAL_EXECUTED)
    assert len(rows) == 1
    assert rows[0]["payload"]["eval_method"] == "longitudinal_pairs"
    assert rows[0]["payload"]["sample_count"] == 15
    assert rows[0]["payload"]["eval_duration_ms"] == 30000.0


# -- record_eval_failed -------------------------------------------------------


def test_record_eval_failed(metrics):
    metrics.record_eval_failed("overall_timeout", "30min exceeded")
    rows = metrics.query_by_event_type(EVENT_EVAL_FAILED)
    assert len(rows) == 1
    assert rows[0]["payload"]["eval_failure_type"] == "overall_timeout"


# -- record_version_rollback --------------------------------------------------


def test_record_version_rollback(metrics):
    metrics.record_version_rollback("v3", "v1", "hash anti-loop hit")
    rows = metrics.query_by_event_type(EVENT_VERSION_ROLLBACK)
    assert len(rows) == 1
    assert rows[0]["payload"]["rollback_from"] == "v3"
    assert rows[0]["payload"]["rollback_to"] == "v1"
    assert rows[0]["payload"]["reason"] == "hash anti-loop hit"


# -- record_blocked_marked ----------------------------------------------------


def test_record_blocked_marked(metrics):
    metrics.record_blocked_marked("evolution_blocked", "evolution", "2026-07-29T00:00:00Z")
    rows = metrics.query_by_event_type(EVENT_BLOCKED_MARKED)
    assert len(rows) == 1
    assert rows[0]["payload"]["blocked_pattern"] == "evolution_blocked"
    assert rows[0]["payload"]["blocked_type"] == "evolution"


# -- record_budget_warning ----------------------------------------------------


def test_record_budget_warning(metrics):
    metrics.record_budget_warning("codex", 0.85, 3.0)
    rows = metrics.query_by_event_type(EVENT_BUDGET_WARNING)
    assert len(rows) == 1
    assert rows[0]["payload"]["specialist_name"] == "codex"
    assert rows[0]["payload"]["usage_percent"] == 0.85
    assert rows[0]["payload"]["remaining_cost_usd"] == 3.0


# -- record_budget_exceeded ---------------------------------------------------


def test_record_budget_exceeded(metrics):
    metrics.record_budget_exceeded("codex", "daily_cost_exceeded", "lead")
    rows = metrics.query_by_event_type(EVENT_BUDGET_EXCEEDED)
    assert len(rows) == 1
    assert rows[0]["payload"]["specialist_name"] == "codex"
    assert rows[0]["payload"]["exceeded_dimension"] == "daily_cost_exceeded"
    assert rows[0]["payload"]["fallback_target"] == "lead"


# -- not ThreadState ----------------------------------------------------------


def test_metrics_not_modify_threadstate(metrics):
    """Metrics write to db, not ThreadState (INV-38)."""
    import ast

    import agent4ml.backend.agents.multiagent.evolution.metrics_l2 as mod

    source = Path(mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    import_names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                import_names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                import_names.append(alias.name)
            if node.module:
                import_names.append(node.module)
    assert not any("ThreadState" in name for name in import_names)
    assert not any("state.types" in name for name in import_names)


# -- persistence --------------------------------------------------------------


def test_persistence_across_sessions(metrics, tmp_path):
    """Process restart - metrics preserved (SQLite)."""
    metrics.record_trigger("periodic", "6h", "default")

    # New instance (simulates restart)
    metrics2 = OrchestrationMetricsL2(db_path=metrics._db_path)
    rows = metrics2.query_by_event_type(EVENT_TRIGGER)
    assert len(rows) == 1
    assert rows[0]["payload"]["trigger_source"] == "periodic"


# -- multiple events ----------------------------------------------------------


def test_multiple_events_same_type(metrics):
    """Multiple events of same type - all recorded."""
    for i in range(5):
        metrics.record_trigger("periodic", f"detail_{i}")
    rows = metrics.query_by_event_type(EVENT_TRIGGER, limit=100)
    assert len(rows) == 5


def test_query_limit(metrics):
    """query_by_event_type respects limit."""
    for _ in range(10):
        metrics.record_trigger("periodic", "")
    rows = metrics.query_by_event_type(EVENT_TRIGGER, limit=3)
    assert len(rows) == 3


def test_query_empty(metrics):
    """No records - returns empty list."""
    rows = metrics.query_by_event_type(EVENT_TRIGGER)
    assert rows == []
