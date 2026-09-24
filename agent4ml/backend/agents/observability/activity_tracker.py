"""RunActivityTracker — track runtime activities for observability.

Each model call, tool call, sandbox command, and specialist invocation
is tracked as an activity with lifecycle (started/finished) and
heartbeat (last_progress_at). Consumers (TUI/CLI) poll get_active()
to display current activity and detect no-progress stalls.

Design: design_docs/45 §3.8.1
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Activity:
    activity_id: str
    kind: str  # "model" | "tool" | "sandbox" | "specialist"
    summary: str
    started_at: float
    last_progress_at: float
    status: str = "running"  # "running" | "ok" | "error" | "cancelled"
    output_size: int = 0
    finished_at: float | None = None
    error: str | None = None

    @property
    def elapsed(self) -> float:
        end = self.finished_at or time.time()
        return end - self.started_at

    @property
    def no_progress_duration(self) -> float:
        return time.time() - self.last_progress_at


class RunActivityTracker:
    """Track runtime activities. Thread-safe via GIL (single-threaded graph execution)."""

    def __init__(
        self,
        heartbeat_interval: float = 10.0,
        no_progress_threshold: float = 180.0,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._heartbeat_interval = heartbeat_interval
        self._no_progress_threshold = no_progress_threshold
        self._on_event = on_event
        self._activities: dict[str, Activity] = {}
        self._completed: list[Activity] = []
        self._last_heartbeat_emit: dict[str, float] = {}

    def start(self, activity_id: str, kind: str, summary: str) -> Activity:
        now = time.time()
        activity = Activity(
            activity_id=activity_id, kind=kind, summary=summary,
            started_at=now, last_progress_at=now,
        )
        self._activities[activity_id] = activity
        self._emit("activity.started", activity)
        return activity

    def finish(
        self, activity_id: str, status: str = "ok",
        error: str | None = None, output_size: int = 0,
    ) -> Activity | None:
        activity = self._activities.pop(activity_id, None)
        if activity is None:
            return None
        activity.status = status
        activity.error = error
        activity.output_size = output_size
        activity.finished_at = time.time()
        self._completed.append(activity)
        self._last_heartbeat_emit.pop(activity_id, None)
        self._emit("activity.finished", activity)
        return activity

    def heartbeat(self, activity_id: str, output_size: int = 0) -> None:
        activity = self._activities.get(activity_id)
        if activity is None:
            return
        activity.last_progress_at = time.time()
        activity.output_size = output_size
        last_emit = self._last_heartbeat_emit.get(activity_id, 0)
        if time.time() - last_emit >= self._heartbeat_interval:
            self._last_heartbeat_emit[activity_id] = time.time()
            self._emit("activity.heartbeat", activity)

    def get_active(self) -> list[Activity]:
        return list(self._activities.values())

    def get_recent_completed(self, limit: int = 8) -> list[Activity]:
        return self._completed[-limit:]

    def get_stale_activities(self) -> list[Activity]:
        return [
            a for a in self._activities.values()
            if a.no_progress_duration >= self._no_progress_threshold
        ]

    def total_elapsed(self) -> float:
        if not self._completed and not self._activities:
            return 0.0
        starts = [a.started_at for a in self._completed + list(self._activities.values())]
        ends = [a.finished_at or time.time() for a in self._completed + list(self._activities.values())]
        return max(ends) - min(starts) if starts else 0.0

    def reset(self) -> None:
        self._activities.clear()
        self._completed.clear()
        self._last_heartbeat_emit.clear()

    def _emit(self, event_type: str, activity: Activity) -> None:
        if self._on_event is None:
            return
        self._on_event({
            "type": event_type,
            "activity_id": activity.activity_id,
            "kind": activity.kind,
            "summary": activity.summary,
            "elapsed": round(activity.elapsed, 1),
            "status": activity.status,
            "output_size": activity.output_size,
        })
