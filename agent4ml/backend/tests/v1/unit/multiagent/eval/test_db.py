"""L3 db schema + L1 metrics decision log CRUD 单测.

测试要点（结合 L1/L2/L3 联动）:
- L3SchemaManager schema 迁移 v2→v3 幂等
- L3 表创建（specialist_decision_log + specialist_decision_log_archive）
- MultiAgentMetricsStore.save_decision_log + get_decision_logs CRUD
- archive_decision_logs 归档（移到 archive 表 + 删除主表）
- 既有 L1/L2 表不破坏
- 联动 L2 FailureCategory + L3 DecisionLogRecord
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from agent4ml.backend.agents.multiagent.eval.db import L3SchemaManager
from agent4ml.backend.agents.multiagent.eval.types import DecisionLogRecord
from agent4ml.backend.agents.multiagent.evolution.types import FailureCategory
from agent4ml.backend.agents.multiagent.metrics import MultiAgentMetricsStore


@pytest.fixture
def temp_db() -> str:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    Path(path).unlink(missing_ok=True)


def _make_record(
    log_id: str = "log1",
    specialist: str = "codex",
    category: FailureCategory | None = FailureCategory.CONTEXT_INSUFFICIENT,
    timestamp: str = "2026-07-28T12:00:00Z",
) -> DecisionLogRecord:
    return DecisionLogRecord(
        log_id=log_id, specialist_name=specialist, task_id="t1",
        goal="g", success_criteria="sc", failure_category=category,
        success_criteria_met=0, lesson_text="need context", timestamp=timestamp,
    )


class TestL3SchemaManager:
    def test_schema_migration_v2_to_v3(self, temp_db):
        """schema 迁移 v2→v3（承接 L2 v1→v2）."""
        manager = L3SchemaManager(db_path=temp_db)
        assert manager.schema_version == 3

    def test_idempotent_reinit(self, temp_db):
        """重复初始化幂等（v3 db 再 init 是 no-op）."""
        L3SchemaManager(db_path=temp_db)
        L3SchemaManager(db_path=temp_db)  # no exception
        assert L3SchemaManager(db_path=temp_db).schema_version == 3

    def test_l3_tables_created(self, temp_db):
        """L3 表创建（specialist_decision_log + archive）."""
        L3SchemaManager(db_path=temp_db)
        import sqlite3
        conn = sqlite3.connect(temp_db)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()
        assert "specialist_decision_log" in tables
        assert "specialist_decision_log_archive" in tables


class TestMultiAgentMetricsStoreDecisionLog:
    def test_save_and_get(self, temp_db):
        """save_decision_log + get_decision_logs CRUD."""
        store = MultiAgentMetricsStore(db_path=temp_db)
        store.save_decision_log(_make_record("log1", "codex"))
        logs = store.get_decision_logs("codex", FailureCategory.CONTEXT_INSUFFICIENT, 10)
        assert len(logs) == 1
        assert logs[0].log_id == "log1"
        assert logs[0].specialist_name == "codex"
        assert logs[0].failure_category == FailureCategory.CONTEXT_INSUFFICIENT

    def test_get_filters_by_specialist(self, temp_db):
        store = MultiAgentMetricsStore(db_path=temp_db)
        store.save_decision_log(_make_record("log1", "codex"))
        store.save_decision_log(_make_record("log2", "claude"))
        logs = store.get_decision_logs("codex", None, 10)
        assert len(logs) == 1
        assert logs[0].specialist_name == "codex"

    def test_get_filters_by_category(self, temp_db):
        store = MultiAgentMetricsStore(db_path=temp_db)
        store.save_decision_log(_make_record("log1", category=FailureCategory.CONTEXT_INSUFFICIENT))
        store.save_decision_log(_make_record("log2", category=FailureCategory.ABILITY_INSUFFICIENT))
        logs = store.get_decision_logs("codex", FailureCategory.CONTEXT_INSUFFICIENT, 10)
        assert len(logs) == 1
        assert logs[0].failure_category == FailureCategory.CONTEXT_INSUFFICIENT

    def test_archive_expired(self, temp_db):
        """archive_decision_logs 归档（移到 archive 表 + 删除主表）."""
        store = MultiAgentMetricsStore(db_path=temp_db)
        # 写一条 100 天前的记录
        store.save_decision_log(_make_record("old", timestamp="2026-04-01T00:00:00Z"))
        # 写一条今天的记录
        store.save_decision_log(_make_record("new", timestamp="2026-07-28T12:00:00Z"))
        # 归档 90 天前的
        count = store.archive_decision_logs(retention_days=90)
        assert count == 1
        # 主表只剩 1 条（new）
        logs = store.get_decision_logs("codex", None, 10)
        assert len(logs) == 1
        assert logs[0].log_id == "new"

    def test_existing_l1_l2_tables_not_broken(self, temp_db):
        """既有 L1/L2 表不破坏（specialist_records / evolution_artifacts / l2_metrics）."""
        store = MultiAgentMetricsStore(db_path=temp_db)
        import sqlite3
        conn = sqlite3.connect(temp_db)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        conn.close()
        assert "specialist_records" in tables  # L1
        assert "evolution_artifacts" in tables  # L2
        assert "l2_metrics" in tables  # L2
        assert "specialist_decision_log" in tables  # L3
