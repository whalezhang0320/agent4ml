"""BudgetGuard — per-specialist budget 三维度记账 + per-day UTC 0 重置 + 超限 fallback lead（R5）。

设计（42 文档 §7.9 + spec.md BudgetGuard Requirement + R5）:
- check_and_record：三维度记账（token + cost_usd + 调用次数），cost_usd 主触发
- get_today_usage：per-day UTC 0 点重置（R5.3）
- 80% 预警写 metrics（不主动通知 LLM，R5.5）
- 超限 fallback 到 lead（通过 tool 返 BudgetExceeded JSON，不污染 system prompt，INV-10/INV-32）
- 持久化 multiagent.db specialist_budget_usage + budget_warnings 表（R5.6）
"""
from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent4ml.backend.agents.journal.events import utc_now_iso
from agent4ml.backend.agents.multiagent.evolution.types import (
    BudgetCheckResult,
    BudgetRemaining,
    CostRecord,
)

_BUDGET_SCHEMA_SQL = """
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
    warning_id     TEXT PRIMARY KEY,
    specialist_name TEXT NOT NULL,
    date           TEXT NOT NULL,
    warning_type   TEXT NOT NULL,
    detail         TEXT,
    timestamp      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_budget_warnings_name_date ON budget_warnings(specialist_name, date);
"""


@dataclass(frozen=True)
class BudgetLimit:
    """per-specialist 单日 budget 上限（R5.1 默认值）.

    per_day_tokens=200000, per_day_cost_usd=$20, per_day_calls=50.
    """

    per_day_tokens: int = 200000
    per_day_cost_usd: float = 20.0
    per_day_calls: int = 50


def _utc_date_str() -> str:
    """UTC 日期字符串 'YYYY-MM-DD'（per-day 重置 key，R5.3）."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class BudgetGuard:
    """per-specialist budget 三维度记账 + per-day 重置 + 超限 fallback lead（R5）.

    INVARIANT:
    - 三维度记账（token + cost_usd + 调用次数），cost_usd 主触发（INV-30，R5.2）
    - per-day UTC 0 点重置（INV-31，R5.3）
    - 超限 fallback lead，通过 tool 返 JSON 通知 LLM（不污染 system prompt，INV-32，R5.4）
    - 80% 预警写 metrics，不主动通知 LLM（INV-33，R5.5）
    - 持久化 multiagent.db（INV-R5.6）
    """

    def __init__(
        self,
        db_path: str = ".agent4ml/multiagent.db",
        limits: dict[str, BudgetLimit] | None = None,
        warning_threshold: float = 0.8,
    ) -> None:
        self._db_path = db_path
        self._limits = limits or {}
        self._warning_threshold = warning_threshold
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
                conn.executescript(_BUDGET_SCHEMA_SQL)
                conn.commit()
            finally:
                conn.close()

    def _get_limit(self, specialist_name: str) -> BudgetLimit:
        """Get specialist budget limit (default fallback)."""
        return self._limits.get(specialist_name, BudgetLimit())

    def check_and_record(
        self,
        specialist_name: str,
        cost: CostRecord,
    ) -> BudgetCheckResult:
        """Check + record (3 dimensions, cost_usd primary trigger).

        Over-limit returns BudgetCheckResult(allowed=False, reason=..., fallback_target="lead").
        80% warning writes budget_warnings table (no LLM notification).
        """
        limit = self._get_limit(specialist_name)
        date_str = _utc_date_str()
        now = utc_now_iso()

        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """SELECT tokens_used, cost_usd_used, calls_used
                        FROM specialist_budget_usage
                        WHERE specialist_name=? AND date=?""",
                    (specialist_name, date_str),
                ).fetchone()
                current_tokens = row[0] if row else 0
                current_cost = row[1] if row else 0.0
                current_calls = row[2] if row else 0

                # 累加后用量
                new_tokens = current_tokens + cost.tokens
                new_cost = current_cost + cost.cost_usd
                new_calls = current_calls + cost.calls

                # 写入用量（UPSERT）
                conn.execute(
                    """INSERT INTO specialist_budget_usage
                        (specialist_name, date, tokens_used, cost_usd_used,
                         calls_used, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(specialist_name, date) DO UPDATE SET
                            tokens_used=excluded.tokens_used,
                            cost_usd_used=excluded.cost_usd_used,
                            calls_used=excluded.calls_used,
                            last_updated=excluded.last_updated""",
                    (specialist_name, date_str, new_tokens, new_cost, new_calls, now),
                )

                # 检查超限（cost_usd 主触发）
                reason: str | None = None
                if new_cost > limit.per_day_cost_usd:
                    reason = "daily_cost_exceeded"
                elif new_tokens > limit.per_day_tokens:
                    reason = "daily_tokens_exceeded"
                elif new_calls > limit.per_day_calls:
                    reason = "daily_calls_exceeded"

                # 80% 预警（cost_usd 主维度）
                old_pct = current_cost / limit.per_day_cost_usd if limit.per_day_cost_usd > 0 else 0.0
                new_pct = new_cost / limit.per_day_cost_usd if limit.per_day_cost_usd > 0 else 0.0
                if old_pct < self._warning_threshold <= new_pct and reason is None:
                    warning_id = f"warn_{specialist_name}_{date_str}_{int(new_pct * 100)}"
                    conn.execute(
                        """INSERT INTO budget_warnings
                            (warning_id, specialist_name, date, warning_type, detail, timestamp)
                            VALUES (?, ?, ?, ?, ?, ?)""",
                        (warning_id, specialist_name, date_str,
                         "approaching_80_percent",
                         f"cost_usd {new_pct:.0%} of limit", now),
                    )

                conn.commit()
            finally:
                conn.close()

        if reason is not None:
            return BudgetCheckResult(
                allowed=False,
                specialist_name=specialist_name,
                reason=reason,
                remaining=BudgetRemaining(
                    tokens=max(0, limit.per_day_tokens - new_tokens),
                    cost_usd=max(0.0, limit.per_day_cost_usd - new_cost),
                    calls=max(0, limit.per_day_calls - new_calls),
                ),
                fallback_target="lead",
            )

        return BudgetCheckResult(
            allowed=True,
            specialist_name=specialist_name,
            reason=None,
            remaining=BudgetRemaining(
                tokens=max(0, limit.per_day_tokens - new_tokens),
                cost_usd=max(0.0, limit.per_day_cost_usd - new_cost),
                calls=max(0, limit.per_day_calls - new_calls),
            ),
            fallback_target="lead",
        )

    def get_today_usage(self, specialist_name: str) -> dict[str, Any]:
        """Per-day UTC 0 reset (R5.3).

        Returns today's usage dict (tokens_used / cost_usd_used / calls_used).
        No record returns all 0.
        """
        date_str = _utc_date_str()
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    """SELECT tokens_used, cost_usd_used, calls_used
                        FROM specialist_budget_usage
                        WHERE specialist_name=? AND date=?""",
                    (specialist_name, date_str),
                ).fetchone()
            finally:
                conn.close()
        if row is None:
            return {"tokens_used": 0, "cost_usd_used": 0.0, "calls_used": 0}
        return {
            "tokens_used": row[0],
            "cost_usd_used": row[1],
            "calls_used": row[2],
        }

    def fallback_target(self, specialist_name: str) -> str:
        """Over-limit fallback target fixed 'lead' (no other specialist, INV-10)."""
        return "lead"

    def get_warnings(self, specialist_name: str, date_str: str | None = None) -> list[dict]:
        """Query 80% warning records (CLI inspect, no push)."""
        date_str = date_str or _utc_date_str()
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(
                    """SELECT warning_id, warning_type, detail, timestamp
                        FROM budget_warnings
                        WHERE specialist_name=? AND date=?
                        ORDER BY timestamp DESC""",
                    (specialist_name, date_str),
                ).fetchall()
            finally:
                conn.close()
        return [
            {
                "warning_id": r[0],
                "warning_type": r[1],
                "detail": r[2] or "",
                "timestamp": r[3],
            }
            for r in rows
        ]
