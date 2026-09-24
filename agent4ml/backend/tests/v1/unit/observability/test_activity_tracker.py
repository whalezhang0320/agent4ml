"""Tests for RunActivityTracker lifecycle and heartbeat."""

from __future__ import annotations

import time

from agent4ml.backend.agents.observability.activity_tracker import (
    Activity,
    RunActivityTracker,
)


class TestActivityLifecycle:
    def test_start_creates_running_activity(self) -> None:
        tracker = RunActivityTracker()
        act = tracker.start("a1", "tool", "bash: ls")
        assert act.status == "running"
        assert act.kind == "tool"
        assert act.summary == "bash: ls"
        assert act in tracker.get_active()

    def test_finish_removes_from_active_and_adds_to_completed(self) -> None:
        tracker = RunActivityTracker()
        tracker.start("a1", "tool", "bash")
        tracker.finish("a1", status="ok", output_size=100)
        assert tracker.get_active() == []
        assert len(tracker.get_recent_completed()) == 1
        assert tracker.get_recent_completed()[0].status == "ok"

    def test_finish_unknown_id_returns_none(self) -> None:
        tracker = RunActivityTracker()
        assert tracker.finish("nonexistent") is None

    def test_finish_with_error_records_error(self) -> None:
        tracker = RunActivityTracker()
        tracker.start("a1", "tool", "bash")
        tracker.finish("a1", status="error", error="permission denied")
        completed = tracker.get_recent_completed()[0]
        assert completed.status == "error"
        assert completed.error == "permission denied"


class TestHeartbeat:
    def test_heartbeat_updates_progress(self) -> None:
        tracker = RunActivityTracker(heartbeat_interval=0.0)
        tracker.start("a1", "tool", "pip install")
        time.sleep(0.01)
        tracker.heartbeat("a1", output_size=500)
        act = tracker.get_active()[0]
        assert act.output_size == 500

    def test_heartbeat_emits_event_at_interval(self) -> None:
        events: list[dict] = []
        tracker = RunActivityTracker(
            heartbeat_interval=0.0, on_event=events.append,
        )
        tracker.start("a1", "tool", "pip install")
        tracker.heartbeat("a1", output_size=100)
        tracker.heartbeat("a1", output_size=200)
        types = [e["type"] for e in events]
        assert "activity.started" in types
        assert types.count("activity.heartbeat") == 2

    def test_heartbeat_respects_interval(self) -> None:
        events: list[dict] = []
        tracker = RunActivityTracker(
            heartbeat_interval=10.0, on_event=events.append,
        )
        tracker.start("a1", "tool", "pip install")
        tracker.heartbeat("a1", output_size=100)
        tracker.heartbeat("a1", output_size=200)
        types = [e["type"] for e in events]
        assert types.count("activity.heartbeat") == 1


class TestStaleDetection:
    def test_stale_activity_detected(self) -> None:
        tracker = RunActivityTracker(no_progress_threshold=0.01)
        tracker.start("a1", "tool", "bash")
        time.sleep(0.02)
        stale = tracker.get_stale_activities()
        assert len(stale) == 1
        assert stale[0].activity_id == "a1"

    def test_non_stale_not_detected(self) -> None:
        tracker = RunActivityTracker(no_progress_threshold=10.0)
        tracker.start("a1", "tool", "bash")
        assert tracker.get_stale_activities() == []


class TestElapsed:
    def test_activity_elapsed(self) -> None:
        tracker = RunActivityTracker()
        act = tracker.start("a1", "tool", "bash")
        time.sleep(0.01)
        assert act.elapsed >= 0.01

    def test_total_elapsed(self) -> None:
        tracker = RunActivityTracker()
        tracker.start("a1", "tool", "bash")
        time.sleep(0.01)
        tracker.finish("a1")
        assert tracker.total_elapsed() >= 0.01

    def test_total_elapsed_empty(self) -> None:
        tracker = RunActivityTracker()
        assert tracker.total_elapsed() == 0.0


class TestReset:
    def test_reset_clears_all(self) -> None:
        tracker = RunActivityTracker()
        tracker.start("a1", "tool", "bash")
        tracker.finish("a1")
        tracker.start("a2", "tool", "ls")
        tracker.reset()
        assert tracker.get_active() == []
        assert tracker.get_recent_completed() == []
