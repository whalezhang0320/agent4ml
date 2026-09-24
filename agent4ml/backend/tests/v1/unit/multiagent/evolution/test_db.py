"""L2 db schema tests - 6 tables + PRAGMA user_version v1->v2 + idempotent + L1 preserved.

Design (spec.md L2 SQLite schema Requirement + 42 doc S13.20):
- 6 tables: evolution_artifacts / evolution_experiments / l2_metrics /
  specialist_budget_usage / budget_warnings / l2_blocked_patterns
- PRAGMA user_version v1 -> v2, idempotent
- Existing L1 tables (specialist_records / specialist_judgments) not broken
"""
from __future__ import annotations

import sqlite3

import pytest

from agent4ml.backend.agents.multiagent.evolution.db import L2SchemaManager


_L1_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS specialist_records (
    specialist_name     TEXT PRIMARY KEY,
    total_selections    INTEGER NOT NULL DEFAULT 0,
    total_invoked       INTEGER NOT NULL DEFAULT 0,
    total_completions   INTEGER NOT NULL DEFAULT 0,
    total_fallbacks     INTEGER NOT NULL DEFAULT 0,
    created_at          TEXT NOT NULL,
    last_updated        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS specialist_judgments (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id              TEXT NOT NULL,
    specialist_name     TEXT NOT NULL,
    success             INTEGER NOT NULL DEFAULT 0,
    duration_seconds    REAL NOT NULL DEFAULT 0.0,
    prompt_tokens       INTEGER NOT NULL DEFAULT 0,
    completion_tokens   INTEGER NOT NULL DEFAULT 0,
    gap_analysis        TEXT NOT NULL DEFAULT '',
    ts                  TEXT NOT NULL
);
"""


def _get_user_version(db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()


def _table_exists(db_path: str, table_name: str) -> bool:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def _list_tables(db_path: str) -> list[str]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


# -- schema migration v1 -> v2 idempotent -------------------------------------


def test_schema_creates_6_tables(tmp_path):
    """L2 schema creates 6 tables."""
    db_path = str(tmp_path / "test.db")
    L2SchemaManager(db_path=db_path)
    assert _table_exists(db_path, "evolution_artifacts")
    assert _table_exists(db_path, "evolution_experiments")
    assert _table_exists(db_path, "l2_metrics")
    assert _table_exists(db_path, "specialist_budget_usage")
    assert _table_exists(db_path, "budget_warnings")
    assert _table_exists(db_path, "l2_blocked_patterns")


def test_user_version_upgraded_to_2(tmp_path):
    """PRAGMA user_version upgraded to 2 (from 0 = empty db)."""
    db_path = str(tmp_path / "test.db")
    manager = L2SchemaManager(db_path=db_path)
    assert _get_user_version(db_path) == manager.schema_version
    assert _get_user_version(db_path) == 2


def test_schema_idempotent_re_run(tmp_path):
    """Re-running on existing v2 db is no-op (idempotent)."""
    db_path = str(tmp_path / "test.db")
    L2SchemaManager(db_path=db_path)
    # Run again
    L2SchemaManager(db_path=db_path)
    # Tables still exist, version still 2
    assert _get_user_version(db_path) == 2
    assert _table_exists(db_path, "evolution_artifacts")


def test_schema_preserves_l1_tables(tmp_path):
    """Existing L1 tables (specialist_records / specialist_judgments) preserved."""
    db_path = str(tmp_path / "test.db")
    # Pre-create L1 schema + data (version 1)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_L1_SCHEMA_SQL)
        conn.execute("PRAGMA user_version = 1")
        conn.execute(
            """INSERT INTO specialist_records
                (specialist_name, total_selections, total_invoked,
                 total_completions, total_fallbacks, created_at, last_updated)
                VALUES ('codex', 5, 5, 4, 1, '2026-07-28', '2026-07-28')"""
        )
        conn.commit()
    finally:
        conn.close()

    # Run L2 schema migration
    L2SchemaManager(db_path=db_path)

    # L1 data preserved
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT specialist_name, total_selections FROM specialist_records WHERE specialist_name='codex'"
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    assert row[0] == "codex"
    assert row[1] == 5

    # Version upgraded to 2
    assert _get_user_version(db_path) == 2


def test_schema_version_property():
    """L2SchemaManager.schema_version returns 2."""
    manager = L2SchemaManager.__new__(L2SchemaManager)
    manager._db_path = ""  # not used for property
    # schema_version is class-level constant
    assert L2SchemaManager.schema_version.fget(manager) == 2


# -- index created ------------------------------------------------------------


def test_indexes_created(tmp_path):
    """Required indexes created."""
    db_path = str(tmp_path / "test.db")
    L2SchemaManager(db_path=db_path)

    conn = sqlite3.connect(db_path)
    try:
        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' ORDER BY name"
        ).fetchall()
        index_names = {r[0] for r in indexes}
    finally:
        conn.close()

    assert "idx_evo_exp_artifact" in index_names
    assert "idx_l2_metrics_event_type" in index_names
    assert "idx_l2_metrics_timestamp" in index_names
    assert "idx_budget_warnings_name_date" in index_names


# -- UNIQUE constraint --------------------------------------------------------


def test_evolution_artifacts_unique_constraint(tmp_path):
    """evolution_artifacts UNIQUE(artifact_type, template_id, version)."""
    db_path = str(tmp_path / "test.db")
    L2SchemaManager(db_path=db_path)

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """INSERT INTO evolution_artifacts
                (artifact_id, artifact_type, version, template_id,
                 payload_json, artifact_hash, rationale, created_at, is_active)
                VALUES ('a1', 'context_summary', 'v1', 'default', '{}', 'h1', '', 'now', 0)"""
        )
        # Duplicate (same artifact_type, template_id, version) should fail
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO evolution_artifacts
                    (artifact_id, artifact_type, version, template_id,
                     payload_json, artifact_hash, rationale, created_at, is_active)
                    VALUES ('a2', 'context_summary', 'v1', 'default', '{}', 'h2', '', 'now', 0)"""
            )
        conn.commit()
    finally:
        conn.close()


def test_specialist_budget_usage_primary_key(tmp_path):
    """specialist_budget_usage PRIMARY KEY (specialist_name, date)."""
    db_path = str(tmp_path / "test.db")
    L2SchemaManager(db_path=db_path)

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """INSERT INTO specialist_budget_usage
                (specialist_name, date, tokens_used, cost_usd_used,
                 calls_used, last_updated)
                VALUES ('codex', '2026-07-28', 100, 0.5, 1, 'now')"""
        )
        # Duplicate (same specialist_name, date) should fail
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO specialist_budget_usage
                    (specialist_name, date, tokens_used, cost_usd_used,
                     calls_used, last_updated)
                    VALUES ('codex', '2026-07-28', 200, 1.0, 2, 'now')"""
            )
        conn.commit()
    finally:
        conn.close()
