"""L3 db schema management - 2 tables + PRAGMA user_version v2->v3.

Design (43 doc §7 + spec.md L3 SQLite schema Requirement):
- 2 tables: specialist_decision_log / specialist_decision_log_archive
- PRAGMA user_version v2 (L2) -> v3 (L3), idempotent
- Same db as L1/L2 metrics (Z3 mode, different tables)
- Existing L1/L2 tables not broken
- L3-7.5 决策 c: 90 天 + 归档（不删除，移到 archive 表）
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

_L3_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS specialist_decision_log (
    log_id                  TEXT PRIMARY KEY,
    specialist_name         TEXT NOT NULL,
    task_id                 TEXT NOT NULL,
    goal                    TEXT NOT NULL,
    success_criteria        TEXT NOT NULL,
    failure_category        TEXT,
    success_criteria_met    INTEGER,
    lesson_text             TEXT,
    timestamp               TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_decision_log_specialist ON specialist_decision_log(specialist_name, timestamp);
CREATE INDEX IF NOT EXISTS idx_decision_log_category ON specialist_decision_log(failure_category, timestamp);

CREATE TABLE IF NOT EXISTS specialist_decision_log_archive (
    log_id                  TEXT PRIMARY KEY,
    specialist_name         TEXT NOT NULL,
    task_id                 TEXT NOT NULL,
    goal                    TEXT NOT NULL,
    success_criteria        TEXT NOT NULL,
    failure_category        TEXT,
    success_criteria_met    INTEGER,
    lesson_text             TEXT,
    timestamp               TEXT NOT NULL,
    archived_at             TEXT NOT NULL
);
"""

_L3_SCHEMA_VERSION = 3


class L3SchemaManager:
    """L3 SQLite schema manager (2 tables + PRAGMA user_version v2->v3).

    Idempotent: re-run on existing v3 db is a no-op.
    Existing L1/L2 tables (specialist_records / evolution_artifacts / l2_metrics) preserved.
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
                conn.executescript(_L3_SCHEMA_SQL)
                if version < _L3_SCHEMA_VERSION:
                    conn.execute(f"PRAGMA user_version = {_L3_SCHEMA_VERSION}")
                conn.commit()
            finally:
                conn.close()

    @property
    def schema_version(self) -> int:
        return _L3_SCHEMA_VERSION
