"""MultiAgentMetricsStore — SQLite metrics for specialist orchestration.

设计（spec.md MultiAgentMetricsStore Requirement + design.md §6）:
- .agent4ml/multiagent.db SQLite（WAL + busy_timeout + threading.Lock）
- specialist_records 表：name + 4 计数器（selections/invoked/completions/fallbacks）
- specialist_judgments 表：run_id + specialist + success + duration + gap
- health_check：completion_rate < threshold AND invoked >= min → degraded
- 不进 ThreadState（design.md §2 metrics 分离）
"""
from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent4ml.backend.agents.journal.events import utc_now_iso

_SCHEMA_VERSION = 3

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS specialist_records (
    specialist_name     TEXT PRIMARY KEY,
    total_selections    INTEGER NOT NULL DEFAULT 0,
    total_invoked       INTEGER NOT NULL DEFAULT 0,
    total_completions   INTEGER NOT NULL DEFAULT 0,
    total_fallbacks     INTEGER NOT NULL DEFAULT 0,
    failure_category    TEXT,
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
    cost_usd            REAL NOT NULL DEFAULT 0.0,
    failure_category    TEXT,
    gap_analysis        TEXT NOT NULL DEFAULT '',
    ts                  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sj_name ON specialist_judgments(specialist_name);
CREATE INDEX IF NOT EXISTS idx_sj_run  ON specialist_judgments(run_id);

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


@dataclass(frozen=True)
class SpecialistMetrics:
    """specialist 聚合指标（从 specialist_records 派生）。"""

    specialist_name: str
    total_selections: int
    total_invoked: int
    total_completions: int
    total_fallbacks: int

    @property
    def completion_rate(self) -> float:
        if self.total_invoked == 0:
            return 0.0
        return self.total_completions / self.total_invoked

    @property
    def fallback_rate(self) -> float:
        if self.total_invoked == 0:
            return 0.0
        return self.total_fallbacks / self.total_invoked


@dataclass(frozen=True)
class SpecialistHealth:
    """specialist 健康状态（health_check 输出）。"""

    specialist_name: str
    completion_rate: float
    total_invoked: int
    degraded: bool


class MultiAgentMetricsStore:
    """SQLite metrics store for multi-agent orchestration.

    INVARIANT:
    - WAL 模式 + busy_timeout=30000 + threading.Lock 保护写
    - PRAGMA user_version 记 schema 版本，首次启动建表
    - 4 计数器 programmatic 打点（record_selection/invoked/completion/fallback）
    - 不进 ThreadState（design.md §2 metrics 分离）
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
                if version < _SCHEMA_VERSION:
                    conn.executescript(_SCHEMA_SQL)
                    conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
                    conn.commit()
            finally:
                conn.close()

    def _upsert_counter(self, specialist_name: str, field: str) -> None:
        now = utc_now_iso()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    f"""INSERT INTO specialist_records (specialist_name, {field}, created_at, last_updated)
                        VALUES (?, 1, ?, ?)
                        ON CONFLICT(specialist_name) DO UPDATE SET
                            {field} = {field} + 1,
                            last_updated = ?""",
                    (specialist_name, now, now, now),
                )
                conn.commit()
            finally:
                conn.close()

    def record_selection(self, specialist_name: str) -> None:
        self._upsert_counter(specialist_name, "total_selections")

    def record_invoked(self, specialist_name: str) -> None:
        self._upsert_counter(specialist_name, "total_invoked")

    def record_completion(self, specialist_name: str) -> None:
        self._upsert_counter(specialist_name, "total_completions")

    def record_fallback(self, specialist_name: str) -> None:
        self._upsert_counter(specialist_name, "total_fallbacks")

    def record_judgment(
        self,
        run_id: str,
        specialist_name: str,
        success: bool,
        duration_seconds: float = 0.0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_usd: float = 0.0,
        failure_category: str | None = None,
        gap_analysis: str = "",
    ) -> None:
        now = utc_now_iso()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """INSERT INTO specialist_judgments
                        (run_id, specialist_name, success, duration_seconds,
                         prompt_tokens, completion_tokens, cost_usd,
                         failure_category, gap_analysis, ts)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        run_id, specialist_name, int(success), duration_seconds,
                        prompt_tokens, completion_tokens, cost_usd,
                        failure_category, gap_analysis, now,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def get_metrics(self, specialist_name: str) -> SpecialistMetrics | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """SELECT specialist_name, total_selections, total_invoked,
                              total_completions, total_fallbacks
                        FROM specialist_records WHERE specialist_name = ?""",
                    (specialist_name,),
                ).fetchone()
            finally:
                conn.close()
        if row is None:
            return None
        return SpecialistMetrics(
            specialist_name=row[0],
            total_selections=row[1],
            total_invoked=row[2],
            total_completions=row[3],
            total_fallbacks=row[4],
        )

    def get_top_specialists(self, limit: int = 10) -> list[SpecialistMetrics]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """SELECT specialist_name, total_selections, total_invoked,
                              total_completions, total_fallbacks
                        FROM specialist_records
                        ORDER BY total_invoked DESC LIMIT ?""",
                    (limit,),
                ).fetchall()
            finally:
                conn.close()
        return [
            SpecialistMetrics(
                specialist_name=r[0], total_selections=r[1], total_invoked=r[2],
                total_completions=r[3], total_fallbacks=r[4],
            )
            for r in rows
        ]

    def health_check(
        self,
        threshold: float = 0.4,
        min_invoked: int = 5,
    ) -> list[SpecialistHealth]:
        """Return degraded specialists (completion_rate < threshold AND invoked >= min)."""
        all_metrics = self.get_top_specialists(limit=100)
        return [
            SpecialistHealth(
                specialist_name=m.specialist_name,
                completion_rate=m.completion_rate,
                total_invoked=m.total_invoked,
                degraded=(m.completion_rate < threshold and m.total_invoked >= min_invoked),
            )
            for m in all_metrics
        ]

    # ── MetricsView Protocol implementation (L2 reads L1 via these methods) ──

    def get_specialist_metrics(
        self, name: str, *, since: float | None = None
    ) -> dict | None:
        """Single specialist aggregated snapshot (JOIN specialist_judgments for cost/latency)."""
        base = self.get_metrics(name)
        if base is None:
            return None
        with self._lock:
            conn = self._connect()
            try:
                if since is not None:
                    since_str = utc_now_iso()
                    rows = conn.execute(
                        """SELECT duration_seconds, prompt_tokens, completion_tokens
                            FROM specialist_judgments
                            WHERE specialist_name=? AND ts >= ?
                            ORDER BY ts DESC LIMIT 20""",
                        (name, since_str),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """SELECT duration_seconds, prompt_tokens, completion_tokens
                            FROM specialist_judgments
                            WHERE specialist_name=?
                            ORDER BY ts DESC LIMIT 20""",
                        (name,),
                    ).fetchall()
            finally:
                conn.close()
        if rows:
            avg_latency = sum(r[0] for r in rows) / len(rows)
            # MVP cost estimate: tokens * $0.00002 (placeholder, real price from config)
            total_tokens = sum(r[1] + r[2] for r in rows)
            avg_cost = total_tokens * 0.00002 / len(rows)
            sample_size = len(rows)
        else:
            avg_latency = 0.0
            avg_cost = 0.0
            sample_size = 0
        return {
            "specialist_name": base.specialist_name,
            "total_selections": base.total_selections,
            "total_invoked": base.total_invoked,
            "total_completions": base.total_completions,
            "total_fallbacks": base.total_fallbacks,
            "completion_rate": base.completion_rate,
            "avg_cost_usd": avg_cost,
            "avg_latency_seconds": avg_latency,
            "sample_size": sample_size,
        }

    def get_global_metrics(self, *, since: float | None = None) -> dict:
        """Global aggregated snapshot across all specialists."""
        all_metrics = self.get_top_specialists(limit=100)
        total_calls = sum(m.total_invoked for m in all_metrics)
        total_selections = sum(m.total_selections for m in all_metrics)
        total_completions = sum(m.total_completions for m in all_metrics)
        total_fallbacks = sum(m.total_fallbacks for m in all_metrics)
        # avg latency/cost across all judgments
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """SELECT AVG(duration_seconds), SUM(prompt_tokens + completion_tokens) * 0.00002
                        FROM specialist_judgments"""
                ).fetchone()
            finally:
                conn.close()
        avg_latency = row[0] if row and row[0] else 0.0
        total_cost = row[1] if row and row[1] else 0.0
        return {
            "total_calls": total_calls,
            "total_cost_usd": total_cost,
            "avg_latency_seconds": avg_latency,
            "total_selections": total_selections,
            "total_completions": total_completions,
            "total_fallbacks": total_fallbacks,
        }

    def get_failure_categories(self, *, since: float | None = None) -> dict:
        """Failure category counts (from specialist_judgments.failure_category)."""
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """SELECT failure_category, COUNT(*)
                        FROM specialist_judgments
                        WHERE failure_category IS NOT NULL
                        GROUP BY failure_category"""
                ).fetchall()
            finally:
                conn.close()
        # Map string category to FailureCategory enum (lazy import to avoid circular)
        from agent4ml.backend.agents.multiagent.evolution.types import FailureCategory
        result: dict = {}
        for cat_str, count in rows:
            try:
                cat = FailureCategory(cat_str)
                result[cat] = count
            except ValueError:
                continue  # unknown category string, skip
        return result

    def get_recent_failures(
        self, *, category, limit: int = 10
    ) -> list:
        """Recent N failures of a given category."""
        cat_str = category.value if hasattr(category, "value") else str(category)
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """SELECT specialist_name, gap_analysis, failure_category, ts
                        FROM specialist_judgments
                        WHERE failure_category=?
                        ORDER BY ts DESC LIMIT ?""",
                    (cat_str, limit),
                ).fetchall()
            finally:
                conn.close()
        from agent4ml.backend.agents.multiagent.evolution.types import FailureRecord
        records: list[FailureRecord] = []
        for r in rows:
            try:
                fc = FailureCategory(r[2])
            except ValueError:
                continue
            records.append(FailureRecord(
                specialist_name=r[0],
                goal="",
                success_criteria="",
                failure_category=fc,
                raw_output_tail=r[1] or "",
                timestamp=r[3] or "",
            ))
        return records

    def list_specialists(self) -> list[str]:
        """All specialist names with records."""
        all_metrics = self.get_top_specialists(limit=100)
        return [m.specialist_name for m in all_metrics]

    def save_decision_log(self, record: Any) -> None:
        """Save decision log record (L3 DecisionLogRecord, duck typing).

        L3-7.1: 跨 run specialist 协作 lessons 累积.
        failure_category 存 enum .value（字符串，避免 import L3 类型）.
        """
        now = utc_now_iso()
        fc_value = record.failure_category.value if record.failure_category else None
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO specialist_decision_log
                       (log_id, specialist_name, task_id, goal, success_criteria,
                        failure_category, success_criteria_met, lesson_text, timestamp)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (record.log_id, record.specialist_name, record.task_id,
                     record.goal, record.success_criteria, fc_value,
                     record.success_criteria_met, record.lesson_text,
                     record.timestamp or now),
                )
                conn.commit()
            finally:
                conn.close()

    def get_decision_logs(
        self, specialist_name: str, failure_category: Any | None, limit: int,
    ) -> list[Any]:
        """Query decision logs (lazy import L3 types to avoid circular dependency).

        Returns list[DecisionLogRecord]. failure_category None = no filter.
        """
        from agent4ml.backend.agents.multiagent.evolution.types import FailureCategory
        from agent4ml.backend.agents.multiagent.eval.types import DecisionLogRecord

        with self._lock:
            conn = self._connect()
            try:
                if failure_category is not None:
                    cursor = conn.execute(
                        """SELECT log_id, specialist_name, task_id, goal, success_criteria,
                                  failure_category, success_criteria_met, lesson_text, timestamp
                           FROM specialist_decision_log
                           WHERE specialist_name=? AND failure_category=?
                           ORDER BY timestamp DESC LIMIT ?""",
                        (specialist_name, failure_category.value, limit),
                    )
                else:
                    cursor = conn.execute(
                        """SELECT log_id, specialist_name, task_id, goal, success_criteria,
                                  failure_category, success_criteria_met, lesson_text, timestamp
                           FROM specialist_decision_log
                           WHERE specialist_name=?
                           ORDER BY timestamp DESC LIMIT ?""",
                        (specialist_name, limit),
                    )
                rows = cursor.fetchall()
            finally:
                conn.close()

        records: list[DecisionLogRecord] = []
        for row in rows:
            fc = FailureCategory(row[5]) if row[5] else None
            records.append(DecisionLogRecord(
                log_id=row[0], specialist_name=row[1], task_id=row[2],
                goal=row[3], success_criteria=row[4], failure_category=fc,
                success_criteria_met=row[6], lesson_text=row[7], timestamp=row[8],
            ))
        return records

    def archive_decision_logs(self, retention_days: int) -> int:
        """Archive expired decision logs (move to archive table + delete main, L3-7.5)."""
        from datetime import datetime, timedelta, timezone
        cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
        now = utc_now_iso()
        with self._lock:
            conn = self._connect()
            try:
                cursor = conn.execute(
                    "SELECT log_id FROM specialist_decision_log WHERE timestamp < ?",
                    (cutoff,),
                )
                expired_ids = [row[0] for row in cursor.fetchall()]
                if not expired_ids:
                    return 0
                placeholders = ",".join("?" * len(expired_ids))
                conn.execute(
                    f"""INSERT INTO specialist_decision_log_archive
                        (log_id, specialist_name, task_id, goal, success_criteria,
                         failure_category, success_criteria_met, lesson_text, timestamp, archived_at)
                        SELECT log_id, specialist_name, task_id, goal, success_criteria,
                               failure_category, success_criteria_met, lesson_text, timestamp, ?
                        FROM specialist_decision_log WHERE log_id IN ({placeholders})""",
                    [now] + expired_ids,
                )
                conn.execute(
                    f"DELETE FROM specialist_decision_log WHERE log_id IN ({placeholders})",
                    expired_ids,
                )
                conn.commit()
                return len(expired_ids)
            finally:
                conn.close()
