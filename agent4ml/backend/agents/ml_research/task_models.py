"""Durable task, workflow node, approval, queue, and event models."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Iterator


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    WAITING_EXTERNAL = "waiting_external"
    RETRY_WAIT = "retry_wait"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class NodeStatus(StrEnum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    WAITING_EXTERNAL = "waiting_external"
    RETRY_WAIT = "retry_wait"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


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
        {
            TaskStatus.QUEUED,
            TaskStatus.WAITING_APPROVAL,
            TaskStatus.WAITING_EXTERNAL,
            TaskStatus.RETRY_WAIT,
            TaskStatus.CANCELLING,
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
        }
    ),
    TaskStatus.WAITING_APPROVAL: frozenset(
        {TaskStatus.QUEUED, TaskStatus.CANCELLED, TaskStatus.FAILED}
    ),
    TaskStatus.WAITING_EXTERNAL: frozenset(
        {
            TaskStatus.QUEUED,
            TaskStatus.RETRY_WAIT,
            TaskStatus.WAITING_APPROVAL,
            TaskStatus.CANCELLING,
            TaskStatus.CANCELLED,
            TaskStatus.FAILED,
        }
    ),
    TaskStatus.RETRY_WAIT: frozenset(
        {TaskStatus.QUEUED, TaskStatus.CANCELLED, TaskStatus.FAILED}
    ),
    TaskStatus.CANCELLING: frozenset({TaskStatus.CANCELLED, TaskStatus.FAILED}),
    TaskStatus.CANCELLED: frozenset(),
    TaskStatus.SUCCEEDED: frozenset(),
    TaskStatus.FAILED: frozenset(),
}

ALLOWED_NODE_TRANSITIONS: dict[NodeStatus, frozenset[NodeStatus]] = {
    NodeStatus.PENDING: frozenset(
        {NodeStatus.QUEUED, NodeStatus.WAITING_APPROVAL, NodeStatus.SKIPPED}
    ),
    NodeStatus.QUEUED: frozenset(
        {NodeStatus.RUNNING, NodeStatus.FAILED, NodeStatus.SKIPPED}
    ),
    NodeStatus.RUNNING: frozenset(
        {
            NodeStatus.SUCCEEDED,
            NodeStatus.RETRY_WAIT,
            NodeStatus.WAITING_APPROVAL,
            NodeStatus.WAITING_EXTERNAL,
            NodeStatus.FAILED,
            NodeStatus.SKIPPED,
        }
    ),
    NodeStatus.WAITING_APPROVAL: frozenset(
        {NodeStatus.QUEUED, NodeStatus.FAILED, NodeStatus.SKIPPED}
    ),
    NodeStatus.WAITING_EXTERNAL: frozenset(
        {
            NodeStatus.SUCCEEDED,
            NodeStatus.RETRY_WAIT,
            NodeStatus.WAITING_APPROVAL,
            NodeStatus.FAILED,
            NodeStatus.SKIPPED,
        }
    ),
    NodeStatus.RETRY_WAIT: frozenset(
        {NodeStatus.QUEUED, NodeStatus.FAILED, NodeStatus.SKIPPED}
    ),
    NodeStatus.SUCCEEDED: frozenset(),
    NodeStatus.FAILED: frozenset(),
    NodeStatus.SKIPPED: frozenset(),
}


class InvalidTaskTransition(ValueError):
    """Raised when a task attempts an illegal state transition."""


class InvalidNodeTransition(ValueError):
    """Raised when a workflow node attempts an illegal state transition."""


class StaleTaskVersion(ValueError):
    """Raised when an optimistic-lock version no longer matches."""


def validate_task_transition(current: TaskStatus, target: TaskStatus) -> None:
    if target not in ALLOWED_TRANSITIONS[current]:
        raise InvalidTaskTransition(f"cannot transition {current.value} -> {target.value}")


def validate_node_transition(current: NodeStatus, target: NodeStatus) -> None:
    if target not in ALLOWED_NODE_TRANSITIONS[current]:
        raise InvalidNodeTransition(f"cannot transition {current.value} -> {target.value}")


@dataclass(frozen=True)
class NodeRecord:
    node_id: str
    status: NodeStatus
    attempt: int = 0
    max_attempts: int = 3
    started_at: str | None = None
    finished_at: str | None = None
    retry_at: str | None = None
    input_hash: str | None = None
    result_refs: tuple[str, ...] = ()
    external_job_id: str | None = None
    failure_class: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        value["result_refs"] = list(self.result_refs)
        return value

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> NodeRecord:
        value = dict(payload)
        value["status"] = NodeStatus(value["status"])
        value["result_refs"] = tuple(value.get("result_refs", ()))
        return cls(**value)


@dataclass(frozen=True)
class ApprovalRecord:
    approval_id: str
    action: str
    reason: str
    payload: dict[str, Any]
    status: ApprovalStatus
    requested_at: str
    resolved_at: str | None = None
    resolved_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        return value

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ApprovalRecord:
        value = dict(payload)
        value["status"] = ApprovalStatus(value["status"])
        return cls(**value)


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
    workflow_type: str | None = None
    workflow_version: int | None = None
    current_node: str | None = None
    nodes: dict[str, NodeRecord] = field(default_factory=dict)
    inputs: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    approval: ApprovalRecord | None = None

    @property
    def is_workflow(self) -> bool:
        return self.workflow_type is not None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["command"] = list(self.command)
        payload["status"] = self.status.value
        payload["nodes"] = {key: node.to_dict() for key, node in self.nodes.items()}
        payload["approval"] = self.approval.to_dict() if self.approval else None
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TaskRecord:
        value = dict(payload)
        value["command"] = tuple(value.get("command", ()))
        value["status"] = TaskStatus(value["status"])
        value["nodes"] = {
            key: NodeRecord.from_dict(node)
            for key, node in value.get("nodes", {}).items()
        }
        approval = value.get("approval")
        value["approval"] = ApprovalRecord.from_dict(approval) if approval else None
        return cls(**value)


@dataclass(frozen=True)
class QueueMessage:
    message_id: str
    task_id: str
    recovered: bool = False
    node_id: str | None = None
    expected_version: int | None = None
    attempt: int | None = None

    # Preserve tuple-style access for callers of the original queue API.
    def __iter__(self) -> Iterator[object]:
        return iter((self.message_id, self.task_id, self.recovered))

    def __getitem__(self, index: int) -> object:
        return (self.message_id, self.task_id, self.recovered)[index]


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
