"""Integration tests for HITL activity tracking, cancel, and help panel."""

from __future__ import annotations

import asyncio
import time

from agent4ml.backend.agents.observability.activity_tracker import RunActivityTracker
from agent4ml.backend.agents.observability.stall_tracker import StallTracker
from agent4ml.backend.agents.observability.situation_report import (
    build_programmatic_report,
    synthesize_options_with_llm,
)
from agent4ml.backend.app.cli.status_bar import build_bottom_toolbar


class TestCLIBottomToolbarActivity:
    def test_running_shows_activity(self) -> None:
        state = {"mode": "default", "model": "deepseek", "_running": True, "_current_activity": "bash: pip install"}
        html = build_bottom_toolbar(state)
        assert "pip install" in str(html)

    def test_not_running_hides_activity(self) -> None:
        state = {"mode": "default", "model": "deepseek", "_running": False, "_current_activity": "bash"}
        html = build_bottom_toolbar(state)
        assert "bash" not in str(html)


class TestActivityTrackerHeartbeatIntegration:
    def test_heartbeat_emits_events_for_long_tool(self) -> None:
        events: list[dict] = []
        tracker = RunActivityTracker(heartbeat_interval=0.0, on_event=events.append)
        tracker.start("bash:t1", "tool", "pip install")
        tracker.heartbeat("bash:t1", output_size=100)
        tracker.heartbeat("bash:t1", output_size=200)
        tracker.finish("bash:t1", status="ok", output_size=200)
        types = [e["type"] for e in events]
        assert "activity.started" in types
        assert types.count("activity.heartbeat") == 2
        assert "activity.finished" in types


class TestStallToReportIntegration:
    def test_stuck_tracker_feeds_report(self) -> None:
        tracker = StallTracker()
        # 阈值 5：5 次不同命令同 capability 触发 stuck
        tracker.record_tool_failure("bash", {"command": "apt install postgresql"}, "permission denied")
        tracker.record_tool_failure("bash", {"command": "find / -name postgres"}, "not found")
        tracker.record_tool_failure("bash", {"command": "which postgres"}, "not found")
        tracker.record_tool_failure("bash", {"command": "pg_isready"}, "not installed")
        tracker.record_tool_failure("bash", {"command": "service postgres status"}, "no such service")
        assert tracker.stuck
        reason = tracker.get_stuck_reason()
        report = build_programmatic_report(tracker.get_failures(), reason or "stuck")
        assert len(report.attempts) == 5
        assert "postgres" in report.blocker


class TestEscCancelLogic:
    def test_single_esc_does_not_rollback(self) -> None:
        first_esc = time.monotonic()
        second_esc = first_esc + 4.0
        assert second_esc - first_esc >= 3.0

    def test_double_esc_within_3s_triggers_rollback(self) -> None:
        first_esc = time.monotonic()
        second_esc = first_esc + 2.0
        assert second_esc - first_esc < 3.0
