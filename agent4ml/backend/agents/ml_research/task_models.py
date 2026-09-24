"""Durable task and event models for the local ML task service."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


TERMINAL_TASK_STATUSES = {
    TaskStatus.CANCELLED,
    TaskStatus.SUCCEEDED,
    TaskStatus.FAILED,
}

ALLOWED_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.QUEUED: frozenset(
        {TaskStatus.RUNNING, TaskStatus.CANCELLED, TaskStatus.FAILED}
    ),
    TaskStatus.RUNNING: frozenset(
        {TaskStatus.CANCELLING, TaskStatus.SUCCEEDED, TaskStatus.FAILED}
    ),
    TaskStatus.CANCELLING: frozenset({TaskStatus.CANCELLED, TaskStatus.FAILED}),
    TaskStatus.CANCELLED: frozenset(),
    TaskStatus.SUCCEEDED: frozenset(),
    TaskStatus.FAILED: frozenset(),
}


class InvalidTaskTransition(ValueError):
    """Raised when a task attempts an illegal state transition."""


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    name: str
    command: tuple[str, ...]
    cwd: str
    run_dir: str
    status: TaskStatus
    created_at: str
    updated_at: str
    version: int = 1
    started_at: str | None = None
    finished_at: str | None = None
    worker_id: str | None = None
    worker_heartbeat_at: str | None = None
    pid: int | None = None
    exit_code: int | None = None
    failure_class: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["command"] = list(self.command)
        payload["status"] = self.status.value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TaskRecord:
        value = dict(payload)
        value["command"] = tuple(value.get("command", ()))
        value["status"] = TaskStatus(value["status"])
        return cls(**value)


@dataclass(frozen=True)
class TaskEvent:
    event_id: str
    task_id: str
    event_type: str
    timestamp: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "task_id": self.task_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "data": self.data,
        }
