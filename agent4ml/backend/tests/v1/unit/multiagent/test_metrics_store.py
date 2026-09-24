"""MultiAgentMetricsStore 单测 — 四计数器 + judgments + health_check + schema 迁移。"""
from __future__ import annotations

import dataclasses

import pytest

from agent4ml.backend.agents.multiagent.metrics import (
    MultiAgentMetricsStore,
    SpecialistHealth,
    SpecialistMetrics,
)


@pytest.fixture
def store(tmp_path):
    return MultiAgentMetricsStore(str(tmp_path / "test.db"))


# ---------------------------------------------------------------------------
# 四计数器打点
# ---------------------------------------------------------------------------


def test_record_selection(store):
    store.record_selection("codex")
    m = store.get_metrics("codex")
    assert m is not None
    assert m.total_selections == 1
    assert m.total_invoked == 0


def test_record_invoked(store):
    store.record_invoked("codex")
    m = store.get_metrics("codex")
    assert m.total_invoked == 1


def test_record_completion(store):
    store.record_completion("codex")
    m = store.get_metrics("codex")
    assert m.total_completions == 1


def test_record_fallback(store):
    store.record_fallback("codex")
    m = store.get_metrics("codex")
    assert m.total_fallbacks == 1


def test_all_four_counters(store):
    store.record_selection("codex")
    store.record_invoked("codex")
    store.record_completion("codex")
    store.record_fallback("codex")

    m = store.get_metrics("codex")
    assert m.total_selections == 1
    assert m.total_invoked == 1
    assert m.total_completions == 1
    assert m.total_fallbacks == 1


def test_counter_accumulates(store):
    store.record_selection("codex")
    store.record_selection("codex")
    store.record_selection("codex")

    m = store.get_metrics("codex")
    assert m.total_selections == 3


def test_get_metrics_missing_returns_none(store):
    assert store.get_metrics("nonexistent") is None


# ---------------------------------------------------------------------------
# completion_rate / fallback_rate
# ---------------------------------------------------------------------------


def test_completion_rate(store):
    store.record_invoked("codex")
    store.record_invoked("codex")
    store.record_invoked("codex")
    store.record_invoked("codex")
    store.record_completion("codex")
    store.record_completion("codex")

    m = store.get_metrics("codex")
    assert m.completion_rate == 0.5


def test_completion_rate_zero_invoked(store):
    store.record_selection("codex")
    m = store.get_metrics("codex")
    assert m.completion_rate == 0.0


# ---------------------------------------------------------------------------
# judgments 记录
# ---------------------------------------------------------------------------


def test_record_judgment(store):
    store.record_judgment(
        run_id="run-1",
        specialist_name="codex",
        success=True,
        duration_seconds=12.5,
        prompt_tokens=100,
        completion_tokens=200,
        gap_analysis="",
    )
    # judgment is stored (no direct read method, but no crash = OK)


def test_record_judgment_failure(store):
    store.record_judgment(
        run_id="run-2",
        specialist_name="claude",
        success=False,
        duration_seconds=5.0,
        gap_analysis="no suggestions found",
    )


# ---------------------------------------------------------------------------
# get_top_specialists
# ---------------------------------------------------------------------------


def test_get_top_specialists(store):
    store.record_invoked("codex")
    store.record_invoked("codex")
    store.record_invoked("claude")

    top = store.get_top_specialists(limit=10)
    assert len(top) == 2
    assert top[0].specialist_name == "codex"
    assert top[0].total_invoked == 2
    assert top[1].specialist_name == "claude"


def test_get_top_specialists_empty(store):
    assert store.get_top_specialists() == []


# ---------------------------------------------------------------------------
# health_check degraded 检测
# ---------------------------------------------------------------------------


def test_health_check_degraded(store):
    # codex: 5 invoked, 1 completion → rate=0.2 < 0.4 → degraded
    for _ in range(5):
        store.record_invoked("codex")
    store.record_completion("codex")

    # claude: 5 invoked, 4 completions → rate=0.8 → not degraded
    for _ in range(5):
        store.record_invoked("claude")
    for _ in range(4):
        store.record_completion("claude")

    health = store.health_check(threshold=0.4, min_invoked=5)
    degraded = [h for h in health if h.degraded]
    healthy = [h for h in health if not h.degraded]

    assert any(h.specialist_name == "codex" and h.degraded for h in health)
    assert any(h.specialist_name == "claude" and not h.degraded for h in health)


def test_health_check_below_min_invoked_not_degraded(store):
    # 2 invoked, 0 completions → rate=0.0 but invoked < 5 → not degraded
    store.record_invoked("codex")
    store.record_invoked("codex")

    health = store.health_check(threshold=0.4, min_invoked=5)
    codex = [h for h in health if h.specialist_name == "codex"]
    assert len(codex) == 1
    assert codex[0].degraded is False


def test_health_check_empty(store):
    assert store.health_check() == []


# ---------------------------------------------------------------------------
# schema 迁移
# ---------------------------------------------------------------------------


def test_schema_migration_creates_tables(tmp_path):
    db_path = str(tmp_path / "multiagent.db")
    store = MultiAgentMetricsStore(db_path)

    import sqlite3
    conn = sqlite3.connect(db_path)
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    conn.close()

    table_names = [t[0] for t in tables]
    assert "specialist_records" in table_names
    assert "specialist_judgments" in table_names


def test_schema_migration_sets_user_version(tmp_path):
    db_path = str(tmp_path / "multiagent.db")
    MultiAgentMetricsStore(db_path)

    import sqlite3
    conn = sqlite3.connect(db_path)
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    conn.close()

    assert version == 3  # L3 schema upgrade (v2 -> v3)


def test_idempotent_init(tmp_path):
    db_path = str(tmp_path / "multiagent.db")
    store1 = MultiAgentMetricsStore(db_path)
    store1.record_selection("codex")

    store2 = MultiAgentMetricsStore(db_path)
    m = store2.get_metrics("codex")
    assert m is not None
    assert m.total_selections == 1


# ---------------------------------------------------------------------------
# frozen dataclasses
# ---------------------------------------------------------------------------


def test_specialist_metrics_frozen():
    m = SpecialistMetrics(
        specialist_name="codex",
        total_selections=1, total_invoked=1,
        total_completions=1, total_fallbacks=0,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.total_selections = 2


def test_specialist_health_frozen():
    h = SpecialistHealth(
        specialist_name="codex",
        completion_rate=0.5,
        total_invoked=10,
        degraded=False,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        h.degraded = True
