"""OrchestrationMetricsL2 - L2 evolution process metrics (11 event types).

Design (42 doc S7.11 + spec.md OrchestrationMetricsL2 Requirement):
- 11 event types: trigger / evolution_start / mutator_call / evolution_failed /
  promotion_decision / eval_executed / eval_failed / version_rollback /
  blocked_marked / budget_warning / budget_exceeded
- Write to multiagent.db l2_metrics table (not ThreadState)
- Each event: event_type + payload_json + timestamp
- Separate from L1 OrchestrationMetrics (specialist call metrics)
"""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from pathlib import Path

from agent4ml.backend.agents.journal.events import utc_now_iso

# 11 L2 event types (R7.1)
EVENT_TRIGGER = "trigger"
EVENT_EVOLUTION_START = "evolution_start"
EVENT_MUTATOR_CALL = "mutator_call"
EVENT_EVOLUTION_FAILED = "evolution_failed"
EVENT_PROMOTION_DECISION = "promotion_decision"
EVENT_EVAL_EXECUTED = "eval_executed"
EVENT_EVAL_FAILED = "eval_failed"
EVENT_VERSION_ROLLBACK = "version_rollback"
EVENT_BLOCKED_MARKED = "blocked_marked"
EVENT_BUDGET_WARNING = "budget_warning"
EVENT_BUDGET_EXCEEDED = "budget_exceeded"

_ALL_EVENT_TYPES = (
    EVENT_TRIGGER,
    EVENT_EVOLUTION_START,
    EVENT_MUTATOR_CALL,
    EVENT_EVOLUTION_FAILED,
    EVENT_PROMOTION_DECISION,
    EVENT_EVAL_EXECUTED,
    EVENT_EVAL_FAILED,
    EVENT_VERSION_ROLLBACK,
    EVENT_BLOCKED_MARKED,
    EVENT_BUDGET_WARNING,
    EVENT_BUDGET_EXCEEDED,
)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS l2_metrics (
    metric_id   TEXT PRIMARY KEY,
    event_type  TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    timestamp   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_l2_metrics_event_type ON l2_metrics(event_type);
CREATE INDEX IF NOT EXISTS idx_l2_metrics_timestamp ON l2_metrics(timestamp);
"""


class OrchestrationMetricsL2:
    """L2 evolution process metrics (11 event types, R7.1).

    INVARIANT:
    - Write to multiagent.db l2_metrics table (not ThreadState, INV-38)
    - 11 event types (R7.1)
    - No active push, user inspects via CLI (INV-39)
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
                conn.executescript(_SCHEMA_SQL)
                conn.commit()
            finally:
                conn.close()

    def record_trigger(
        self,
        trigger_source: str,
        trigger_detail: str = "",
        profile: str = "default",
    ) -> None:
        """Record trigger event."""
        self._record(EVENT_TRIGGER, {
            "trigger_source": trigger_source,
            "trigger_detail": trigger_detail,
            "profile": profile,
        })

    def record_evolution_start(
        self,
        experiment_id: str,
        artifact_type: str,
        from_version: str = "",
    ) -> None:
        """Record evolution start event."""
        self._record(EVENT_EVOLUTION_START, {
            "experiment_id": experiment_id,
            "artifact_type": artifact_type,
            "from_version": from_version,
        })

    def record_mutator_call(
        self,
        llm_model: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float = 0.0,
        retries: int = 0,
    ) -> None:
        """Record EvolutionMutator LLM call."""
        self._record(EVENT_MUTATOR_CALL, {
            "llm_model": llm_model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
            "retries": retries,
        })

    def record_evolution_failed(
        self,
        failure_type: str,
        experiment_id: str = "",
        detail: str = "",
    ) -> None:
        """Record evolution failure (llm_timeout / json_parse / schema_mismatch / illegal_field)."""
        self._record(EVENT_EVOLUTION_FAILED, {
            "failure_type": failure_type,
            "experiment_id": experiment_id,
            "detail": detail,
        })

    def record_promotion_decision(
        self,
        decision: str,
        candidate_score: float = 0.0,
        baseline_score: float = 0.0,
        ci_low: float = 0.0,
        ci_high: float = 0.0,
    ) -> None:
        """Record PromotionGate decision (accept / reject)."""
        self._record(EVENT_PROMOTION_DECISION, {
            "decision": decision,
            "candidate_score": candidate_score,
            "baseline_score": baseline_score,
            "ci_low": ci_low,
            "ci_high": ci_high,
        })

    def record_eval_executed(
        self,
        eval_method: str = "",
        sample_count: int = 0,
        eval_duration_ms: float = 0.0,
        eval_skipped_count: int = 0,
    ) -> None:
        """Record eval execution."""
        self._record(EVENT_EVAL_EXECUTED, {
            "eval_method": eval_method,
            "sample_count": sample_count,
            "eval_duration_ms": eval_duration_ms,
            "eval_skipped_count": eval_skipped_count,
        })

    def record_eval_failed(
        self,
        eval_failure_type: str,
        detail: str = "",
    ) -> None:
        """Record eval failure (task_timeout / sandbox_error / overall_timeout)."""
        self._record(EVENT_EVAL_FAILED, {
            "eval_failure_type": eval_failure_type,
            "detail": detail,
        })

    def record_version_rollback(
        self,
        rollback_from: str = "",
        rollback_to: str = "",
        reason: str = "",
    ) -> None:
        """Record version rollback."""
        self._record(EVENT_VERSION_ROLLBACK, {
            "rollback_from": rollback_from,
            "rollback_to": rollback_to,
            "reason": reason,
        })

    def record_blocked_marked(
        self,
        blocked_pattern: str,
        blocked_type: str = "evolution",
        auto_release_at: str = "",
    ) -> None:
        """Record blocked marking (evolution_blocked / eval_blocked)."""
        self._record(EVENT_BLOCKED_MARKED, {
            "blocked_pattern": blocked_pattern,
            "blocked_type": blocked_type,
            "auto_release_at": auto_release_at,
        })

    def record_budget_warning(
        self,
        specialist_name: str,
        usage_percent: float = 0.0,
        remaining_cost_usd: float = 0.0,
    ) -> None:
        """Record budget 80% warning."""
        self._record(EVENT_BUDGET_WARNING, {
            "specialist_name": specialist_name,
            "usage_percent": usage_percent,
            "remaining_cost_usd": remaining_cost_usd,
        })

    def record_budget_exceeded(
        self,
        specialist_name: str,
        exceeded_dimension: str = "cost_usd",
        fallback_target: str = "lead",
    ) -> None:
        """Record budget exceeded + fallback."""
        self._record(EVENT_BUDGET_EXCEEDED, {
            "specialist_name": specialist_name,
            "exceeded_dimension": exceeded_dimension,
            "fallback_target": fallback_target,
        })

    def _record(self, event_type: str, payload: dict) -> None:
        """Write event to l2_metrics table (not ThreadState, INV-38)."""
        metric_id = f"m2_{uuid.uuid4().hex[:12]}"
        payload_json = json.dumps(payload, sort_keys=True)
        now = utc_now_iso()
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    """INSERT INTO l2_metrics (metric_id, event_type, payload_json, timestamp)
                        VALUES (?, ?, ?, ?)""",
                    (metric_id, event_type, payload_json, now),
                )
                conn.commit()
            finally:
                conn.close()

    def query_by_event_type(self, event_type: str, limit: int = 100) -> list[dict]:
        """Query metrics by event_type (CLI inspect, INV-39)."""
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """SELECT metric_id, event_type, payload_json, timestamp
                        FROM l2_metrics WHERE event_type=? ORDER BY timestamp DESC LIMIT ?""",
                    (event_type, limit),
                ).fetchall()
            finally:
                conn.close()
        return [
            {
                "metric_id": r[0],
                "event_type": r[1],
                "payload": json.loads(r[2]),
                "timestamp": r[3],
            }
            for r in rows
        ]

    @property
    def event_types(self) -> tuple[str, ...]:
        """11 L2 event types (R7.1)."""
        return _ALL_EVENT_TYPES
