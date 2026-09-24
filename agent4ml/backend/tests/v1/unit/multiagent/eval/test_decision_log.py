"""L3 DecisionLog Writer + Reader 单测.

测试要点（结合 L2 FailureCategory 联动）:
- write_async 异步不阻塞（fire-and-forget）
- write_async 实际写入 store（等待 executor 完成）
- get_recent_lessons 过滤 + limit
- archive_expired 调 store.archive_decision_logs
- mock _DecisionLogStore 验证调用
- shutdown 等待 pending 写完成
"""
from __future__ import annotations

import threading
import time

import pytest

from agent4ml.backend.agents.multiagent.eval.decision_log import (
    DecisionLogReader,
    DecisionLogWriter,
)
from agent4ml.backend.agents.multiagent.eval.types import DecisionLogRecord
from agent4ml.backend.agents.multiagent.evolution.types import FailureCategory


class _MockStore:
    """Mock _DecisionLogStore 实现."""

    def __init__(self) -> None:
        self.saved: list[DecisionLogRecord] = []
        self._lock = threading.Lock()
        self.archived_count = 0
        self.archive_retention_days = 0

    def save_decision_log(self, record: DecisionLogRecord) -> None:
        with self._lock:
            self.saved.append(record)

    def get_decision_logs(
        self,
        specialist_name: str,
        failure_category: FailureCategory | None,
        limit: int,
    ) -> list[DecisionLogRecord]:
        result = []
        for r in self.saved:
            if r.specialist_name != specialist_name:
                continue
            if failure_category is not None and r.failure_category != failure_category:
                continue
            result.append(r)
        return result[:limit]

    def archive_decision_logs(self, retention_days: int) -> int:
        self.archived_count = len(self.saved)
        self.archive_retention_days = retention_days
        return self.archived_count


def _make_record(
    log_id: str = "log1",
    specialist: str = "codex",
    category: FailureCategory | None = FailureCategory.CONTEXT_INSUFFICIENT,
) -> DecisionLogRecord:
    return DecisionLogRecord(
        log_id=log_id,
        specialist_name=specialist,
        task_id="t1",
        goal="g",
        success_criteria="sc",
        failure_category=category,
        success_criteria_met=0,
        lesson_text="need more context",
        timestamp="2026-07-28T12:00:00Z",
    )


class TestDecisionLogWriter:
    def test_write_async_does_not_block(self):
        """write_async 异步不阻塞（fire-and-forget）."""
        store = _MockStore()
        writer = DecisionLogWriter(store)
        start = time.time()
        writer.write_async(_make_record())
        elapsed = time.time() - start
        assert elapsed < 0.1  # 异步不阻塞
        writer.shutdown()

    def test_write_async_persists_to_store(self):
        """write_async 实际写入 store（等待 executor 完成）."""
        store = _MockStore()
        writer = DecisionLogWriter(store)
        writer.write_async(_make_record("log1"))
        writer.write_async(_make_record("log2"))
        writer.shutdown()  # 等待 pending 写完成
        assert len(store.saved) == 2
        assert store.saved[0].log_id == "log1"
        assert store.saved[1].log_id == "log2"

    def test_shutdown_waits_for_pending(self):
        """shutdown 等待 pending 写完成."""
        store = _MockStore()
        writer = DecisionLogWriter(store)
        for i in range(5):
            writer.write_async(_make_record(f"log{i}"))
        writer.shutdown()
        assert len(store.saved) == 5


class TestDecisionLogReader:
    def test_get_recent_lessons_filters_by_specialist(self):
        store = _MockStore()
        store.saved = [
            _make_record("log1", specialist="codex"),
            _make_record("log2", specialist="claude"),
            _make_record("log3", specialist="codex"),
        ]
        reader = DecisionLogReader(store)
        lessons = reader.get_recent_lessons("codex", FailureCategory.CONTEXT_INSUFFICIENT)
        assert len(lessons) == 2
        assert all(r.specialist_name == "codex" for r in lessons)

    def test_get_recent_lessons_filters_by_category(self):
        store = _MockStore()
        store.saved = [
            _make_record("log1", category=FailureCategory.CONTEXT_INSUFFICIENT),
            _make_record("log2", category=FailureCategory.ABILITY_INSUFFICIENT),
        ]
        reader = DecisionLogReader(store)
        lessons = reader.get_recent_lessons("codex", FailureCategory.CONTEXT_INSUFFICIENT)
        assert len(lessons) == 1
        assert lessons[0].failure_category == FailureCategory.CONTEXT_INSUFFICIENT

    def test_get_recent_lessons_respects_limit(self):
        store = _MockStore()
        store.saved = [_make_record(f"log{i}") for i in range(10)]
        reader = DecisionLogReader(store)
        lessons = reader.get_recent_lessons(
            "codex", FailureCategory.CONTEXT_INSUFFICIENT, limit=3,
        )
        assert len(lessons) == 3

    def test_archive_expired_calls_store(self):
        store = _MockStore()
        store.saved = [_make_record(f"log{i}") for i in range(5)]
        reader = DecisionLogReader(store)
        count = reader.archive_expired(retention_days=90)
        assert count == 5
        assert store.archive_retention_days == 90
