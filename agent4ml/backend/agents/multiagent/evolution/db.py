"""L2 db schema management - 6 tables + PRAGMA user_version v1->v2.

Design (42 doc S13.20 + spec.md L2 SQLite schema Requirement):
- 6 tables: evolution_artifacts / evolution_experiments / l2_metrics /
  specialist_budget_usage / budget_warnings / l2_blocked_patterns
- PRAGMA user_version v1 (L1) -> v2 (L2), idempotent
- Same db as L1 metrics (Z3 mode, different tables)
- Existing L1 tables (specialist_records / specialist_judgments) not broken
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

_L2_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS evolution_artifacts (
    artifact_id     TEXT PRIMARY KEY,
    artifact_type   TEXT NOT NULL,
    version         TEXT NOT NULL,
    template_id     TEXT NOT NULL,
    payload_json    TEXT NOT NULL,
    artifact_hash   TEXT NOT NULL,
    rationale       TEXT,
    created_at      TEXT NOT NULL,
    is_active       INTEGER DEFAULT 0,
    UNIQUE(artifact_type, template_id, version)
);

CREATE TABLE IF NOT EXISTS evolution_experiments (
    experiment_id   TEXT PRIMARY KEY,
    artifact_id      TEXT NOT NULL REFERENCES evolution_artifacts(artifact_id),
    from_artifact_id TEXT,
    trigger_source   TEXT NOT NULL,
    trigger_detail   TEXT,
    eval_method      TEXT,
    eval_result_json TEXT,
    decision         TEXT NOT NULL,
    timestamp        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_evo_exp_artifact ON evolution_experiments(artifact_id);

CREATE TABLE IF NOT EXISTS l2_metrics (
    metric_id   TEXT PRIMARY KEY,
    event_type  TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    timestamp   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_l2_metrics_event_type ON l2_metrics(event_type);
CREATE INDEX IF NOT EXISTS idx_l2_metrics_timestamp ON l2_metrics(timestamp);

CREATE TABLE IF NOT EXISTS specialist_budget_usage (
    specialist_name TEXT NOT NULL,
    date            TEXT NOT NULL,
    tokens_used     INTEGER DEFAULT 0,
    cost_usd_used   REAL DEFAULT 0.0,
    calls_used      INTEGER DEFAULT 0,
    last_updated    TEXT NOT NULL,
    PRIMARY KEY (specialist_name, date)
);

CREATE TABLE IF NOT EXISTS budget_warnings (
    warning_id      TEXT PRIMARY KEY,
    specialist_name TEXT NOT NULL,
    date            TEXT NOT NULL,
    warning_type    TEXT NOT NULL,
    detail          TEXT,
    timestamp       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_budget_warnings_name_date ON budget_warnings(specialist_name, date);

CREATE TABLE IF NOT EXISTS l2_blocked_patterns (
    blocked_id       TEXT PRIMARY KEY,
    blocked_type     TEXT NOT NULL,
    pattern_key      TEXT NOT NULL,
    reason           TEXT,
    blocked_at       TEXT NOT NULL,
    auto_release_at  TEXT NOT NULL,
    released         INTEGER DEFAULT 0
);
"""

_L2_SCHEMA_VERSION = 2


class L2SchemaManager:
    """L2 SQLite schema manager (6 tables + PRAGMA user_version v1->v2).

    Idempotent: re-run on existing v2 db is a no-op.
    Existing L1 tables (specialist_records / specialist_judgments) preserved.
    """

    def __init__(self, db_path: str = ".agent4ml/multiagent.db") -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            conn = self._connect()
            try:
                version = conn.execute("PRAGMA user_version").fetchone()[0]
                # Always CREATE IF NOT EXISTS (idempotent)
                conn.executescript(_L2_SCHEMA_SQL)
                # Upgrade PRAGMA user_version v1 -> v2 (only if below L2 version)
                if version < _L2_SCHEMA_VERSION:
                    conn.execute(f"PRAGMA user_version = {_L2_SCHEMA_VERSION}")
                conn.commit()
            finally:
                conn.close()

    @property
    def schema_version(self) -> int:
        return _L2_SCHEMA_VERSION
