"""Storage protocol plus in-memory and Redis implementations for ML tasks."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import defaultdict, deque
from dataclasses import replace
from typing import Any, Protocol

from agent4ml.backend.agents.ml_research.task_models import (
    ApprovalRecord,
    ApprovalStatus,
    InvalidNodeTransition,
    InvalidTaskTransition,
    NodeRecord,
    NodeStatus,
    QueueMessage,
    StaleTaskVersion,
    TaskEvent,
    TaskRecord,
    TaskStatus,
    utc_now_iso,
    validate_node_transition,
    validate_task_transition,
)
from agent4ml.backend.agents.ml_research.workflow import NodeResult


class TaskNotFoundError(KeyError):
    pass


class TaskStore(Protocol):
    def create(self, task: TaskRecord) -> TaskRecord: ...
    def submit(self, task: TaskRecord) -> str: ...
    def get(self, task_id: str) -> TaskRecord | None: ...
    def transition(
        self, task_id: str, target: TaskStatus, **changes: Any
    ) -> TaskRecord: ...
    def update(self, task_id: str, **changes: Any) -> TaskRecord: ...
    def enqueue(
        self,
        task_id: str,
        node_id: str | None = None,
        expected_version: int | None = None,
        attempt: int | None = None,
    ) -> str: ...
    def claim(
        self, consumer: str, block_ms: int = 1000
    ) -> QueueMessage | None: ...
    def touch_claim(self, message_id: str, consumer: str) -> None: ...
    def heartbeat_node(
        self, task_id: str, message_id: str, consumer: str
    ) -> None: ...
    def ack(self, message_id: str) -> None: ...
    def request_cancel(self, task_id: str) -> TaskRecord: ...
    def cancel_requested(self, task_id: str) -> bool: ...
    def mark_node_running(
        self,
        task_id: str,
        node_id: str,
        *,
        expected_version: int,
        attempt: int,
        worker_id: str | None = None,
    ) -> TaskRecord: ...
    def commit_node_result(
        self,
        task_id: str,
        node_id: str,
        result: NodeResult,
        next_node: str | None,
        expected_version: int,
        approval: ApprovalRecord | None = None,
    ) -> TaskRecord: ...
    def mark_node_waiting_external(
        self, task_id: str, node_id: str, result: NodeResult, *, expected_version: int
    ) -> TaskRecord: ...
    def fail_node(
        self, task_id: str, node_id: str, result: NodeResult, *, expected_version: int
    ) -> TaskRecord: ...
    def schedule_retry(
        self, task_id: str, node_id: str, result: NodeResult, *, expected_version: int
    ) -> TaskRecord: ...
    def request_node_approval(
        self,
        task_id: str,
        node_id: str,
        approval: ApprovalRecord,
        *,
        expected_version: int,
        result: NodeResult | None = None,
    ) -> TaskRecord: ...
    def resolve_approval(
        self,
        task_id: str,
        approval_id: str,
        *,
        approved: bool,
        expected_version: int,
        resolved_by: str | None = None,
    ) -> TaskRecord: ...
    def retry_failed_node(self, task_id: str, *, expected_version: int) -> TaskRecord: ...
    def waiting_external(self) -> list[TaskRecord]: ...
    def publish(self, task_id: str, event_type: str, data: dict[str, Any]) -> TaskEvent: ...
    def read_events(
        self, task_id: str, after_id: str = "0-0", block_ms: int = 0, count: int = 100
    ) -> list[TaskEvent]: ...
    def close(self) -> None: ...
    def ping(self) -> bool: ...


def _transition_record(
    task: TaskRecord, target: TaskStatus, changes: dict[str, Any]
) -> TaskRecord:
    validate_task_transition(task.status, target)
    now = utc_now_iso()
    defaults: dict[str, Any] = {"updated_at": now, "version": task.version + 1}
    if target is TaskStatus.RUNNING:
        defaults["started_at"] = task.started_at or now
    if target in {TaskStatus.CANCELLED, TaskStatus.SUCCEEDED, TaskStatus.FAILED}:
        defaults["finished_at"] = now
    defaults.update(changes)
    return replace(task, status=target, **defaults)


def _request_cancel_record(task: TaskRecord) -> TaskRecord:
    if task.status in {TaskStatus.RUNNING, TaskStatus.WAITING_EXTERNAL}:
        return _transition_record(task, TaskStatus.CANCELLING, {})
    if task.status in {
        TaskStatus.QUEUED,
        TaskStatus.WAITING_APPROVAL,
        TaskStatus.RETRY_WAIT,
    }:
        now = utc_now_iso()
        nodes = dict(task.nodes)
        if task.current_node and task.current_node in nodes:
            node = nodes[task.current_node]
            try:
                validate_node_transition(node.status, NodeStatus.SKIPPED)
            except InvalidNodeTransition:
                pass
            else:
                nodes[node.node_id] = replace(
                    node,
                    status=NodeStatus.SKIPPED,
                    finished_at=now,
                    error="task cancelled",
                )
        approval = task.approval
        if approval is not None and approval.status is ApprovalStatus.PENDING:
            approval = replace(
                approval,
                status=ApprovalStatus.REJECTED,
                resolved_at=now,
                resolved_by="system:task_cancelled",
            )
        return _transition_record(
            task, TaskStatus.CANCELLED, {"nodes": nodes, "approval": approval}
        )
    return task


def _check_workflow_node(
    task: TaskRecord,
    node_id: str,
    expected_version: int,
    allowed_statuses: set[NodeStatus],
) -> NodeRecord:
    if task.version != expected_version:
        raise StaleTaskVersion(
            f"expected task version {expected_version}, found {task.version}"
        )
    if task.current_node != node_id:
        raise InvalidTaskTransition(
            f"node {node_id} is not current node {task.current_node}"
        )
    try:
        node = task.nodes[node_id]
    except KeyError as exc:
        raise InvalidTaskTransition(f"unknown node: {node_id}") from exc
    if node.status not in allowed_statuses:
        expected = ", ".join(sorted(status.value for status in allowed_statuses))
        raise InvalidTaskTransition(
            f"node {node_id} is {node.status.value}, expected one of: {expected}"
        )
    return node


def _replace_node(task: TaskRecord, node: NodeRecord, **changes: Any) -> TaskRecord:
    nodes = dict(task.nodes)
    nodes[node.node_id] = node
    return replace(
        task,
        nodes=nodes,
        updated_at=utc_now_iso(),
        version=task.version + 1,
        **changes,
    )


def _mark_node_running_record(
    task: TaskRecord, node_id: str, expected_version: int, attempt: int, worker_id: str | None
) -> TaskRecord:
    node = _check_workflow_node(
        task, node_id, expected_version, {NodeStatus.QUEUED}
    )
    if node.attempt != attempt:
        raise StaleTaskVersion(
            f"expected node attempt {attempt}, found {node.attempt}"
        )
    validate_task_transition(task.status, TaskStatus.RUNNING)
    validate_node_transition(node.status, NodeStatus.RUNNING)
    now = utc_now_iso()
    input_hash = hashlib.sha256(
        json.dumps(
            {
                "node_id": node_id,
                "workflow_version": task.workflow_version,
                "inputs": task.inputs,
                "artifacts": task.artifacts,
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    running_node = replace(
        node,
        status=NodeStatus.RUNNING,
        started_at=now,
        finished_at=None,
        retry_at=None,
        failure_class=None,
        error=None,
        input_hash=input_hash,
    )
    return _replace_node(
        task,
        running_node,
        status=TaskStatus.RUNNING,
        started_at=task.started_at or now,
        worker_id=worker_id,
        worker_heartbeat_at=now,
        finished_at=None,
        failure_class=None,
        error=None,
    )


def _commit_node_result_record(
    task: TaskRecord,
    node_id: str,
    result: NodeResult,
    next_node_id: str | None,
    expected_version: int,
    approval: ApprovalRecord | None,
) -> tuple[TaskRecord, tuple[str, int] | None]:
    node = _check_workflow_node(
        task,
        node_id,
        expected_version,
        {NodeStatus.RUNNING, NodeStatus.WAITING_EXTERNAL},
    )
    if result.outcome != "succeeded":
        raise ValueError("commit_node_result requires a succeeded NodeResult")
    validate_node_transition(node.status, NodeStatus.SUCCEEDED)
    now = utc_now_iso()
    refs = tuple(dict.fromkeys((*node.result_refs, *result.artifacts.values())))
    succeeded = replace(
        node,
        status=NodeStatus.SUCCEEDED,
        finished_at=now,
        result_refs=refs,
        error=None,
        failure_class=None,
    )
    nodes = dict(task.nodes)
    nodes[node_id] = succeeded
    artifacts = {**task.artifacts, **result.artifacts}
    if next_node_id is None:
        validate_task_transition(task.status, TaskStatus.SUCCEEDED)
        updated = replace(
            task,
            nodes=nodes,
            artifacts=artifacts,
            current_node=None,
            status=TaskStatus.SUCCEEDED,
            approval=None,
            finished_at=now,
            updated_at=now,
            version=task.version + 1,
        )
        return updated, None

    try:
        next_node = nodes[next_node_id]
    except KeyError as exc:
        raise InvalidTaskTransition(f"unknown next node: {next_node_id}") from exc
    if next_node.status is not NodeStatus.PENDING:
        raise InvalidTaskTransition(
            f"next node {next_node_id} is not pending: {next_node.status.value}"
        )
    if approval is not None:
        validate_task_transition(task.status, TaskStatus.WAITING_APPROVAL)
        validate_node_transition(next_node.status, NodeStatus.WAITING_APPROVAL)
        nodes[next_node_id] = replace(
            next_node, status=NodeStatus.WAITING_APPROVAL
        )
        target = TaskStatus.WAITING_APPROVAL
        queue_item = None
    else:
        validate_task_transition(task.status, TaskStatus.QUEUED)
        validate_node_transition(next_node.status, NodeStatus.QUEUED)
        nodes[next_node_id] = replace(
            next_node, status=NodeStatus.QUEUED, attempt=next_node.attempt + 1
        )
        target = TaskStatus.QUEUED
        queue_item = (next_node_id, nodes[next_node_id].attempt)
    updated = replace(
        task,
        nodes=nodes,
        artifacts=artifacts,
        current_node=next_node_id,
        status=target,
        approval=approval,
        worker_id=None,
        worker_heartbeat_at=None,
        pid=None,
        updated_at=now,
        version=task.version + 1,
    )
    return updated, queue_item


def _waiting_external_record(
    task: TaskRecord, node_id: str, result: NodeResult, expected_version: int
) -> TaskRecord:
    node = _check_workflow_node(
        task, node_id, expected_version, {NodeStatus.RUNNING}
    )
    if result.outcome != "waiting_external" or not result.external_job_id:
        raise ValueError("waiting_external requires an external_job_id")
    validate_task_transition(task.status, TaskStatus.WAITING_EXTERNAL)
    validate_node_transition(node.status, NodeStatus.WAITING_EXTERNAL)
    waiting = replace(
        node,
        status=NodeStatus.WAITING_EXTERNAL,
        external_job_id=result.external_job_id,
        result_refs=tuple(
            dict.fromkeys((*node.result_refs, *result.artifacts.values()))
        ),
    )
    return _replace_node(
        task,
        waiting,
        status=TaskStatus.WAITING_EXTERNAL,
        artifacts={**task.artifacts, **result.artifacts},
        worker_id=None,
        worker_heartbeat_at=None,
        pid=None,
    )


def _fail_node_record(
    task: TaskRecord, node_id: str, result: NodeResult, expected_version: int
) -> TaskRecord:
    node = _check_workflow_node(
        task,
        node_id,
        expected_version,
        {NodeStatus.RUNNING, NodeStatus.WAITING_EXTERNAL},
    )
    validate_task_transition(task.status, TaskStatus.FAILED)
    validate_node_transition(node.status, NodeStatus.FAILED)
    now = utc_now_iso()
    failed = replace(
        node,
        status=NodeStatus.FAILED,
        finished_at=now,
        failure_class=result.failure_class,
        error=result.error,
    )
    return _replace_node(
        task,
        failed,
        status=TaskStatus.FAILED,
        failure_class=result.failure_class,
        error=result.error,
        finished_at=now,
        worker_id=None,
        worker_heartbeat_at=None,
        pid=None,
    )


def _queue_retry_record(
    task: TaskRecord, node_id: str, result: NodeResult, expected_version: int
) -> tuple[TaskRecord, tuple[str, int]]:
    node = _check_workflow_node(
        task,
        node_id,
        expected_version,
        {NodeStatus.RUNNING, NodeStatus.WAITING_EXTERNAL},
    )
    if node.attempt >= node.max_attempts:
        return _fail_node_record(task, node_id, result, expected_version), (
            "",
            0,
        )
    # Phase one uses immediate retry while preserving the retry decision as an event.
    validate_task_transition(task.status, TaskStatus.QUEUED)
    queued = replace(
        node,
        status=NodeStatus.QUEUED,
        attempt=node.attempt + 1,
        started_at=None,
        finished_at=None,
        retry_at=utc_now_iso(),
        failure_class=result.failure_class,
        error=result.error,
    )
    updated = _replace_node(
        task,
        queued,
        status=TaskStatus.QUEUED,
        worker_id=None,
        worker_heartbeat_at=None,
        pid=None,
    )
    return updated, (node_id, queued.attempt)


def _request_node_approval_record(
    task: TaskRecord,
    node_id: str,
    approval: ApprovalRecord,
    expected_version: int,
    result: NodeResult | None,
) -> TaskRecord:
    node = _check_workflow_node(
        task, node_id, expected_version, {NodeStatus.RUNNING, NodeStatus.WAITING_EXTERNAL}
    )
    validate_task_transition(task.status, TaskStatus.WAITING_APPROVAL)
    validate_node_transition(node.status, NodeStatus.WAITING_APPROVAL)
    waiting = replace(
        node,
        status=NodeStatus.WAITING_APPROVAL,
        failure_class=result.failure_class if result else node.failure_class,
        error=result.error if result else node.error,
    )
    return _replace_node(
        task,
        waiting,
        status=TaskStatus.WAITING_APPROVAL,
        approval=approval,
        worker_id=None,
        worker_heartbeat_at=None,
        pid=None,
    )


def _resolve_approval_record(
    task: TaskRecord,
    approval_id: str,
    approved: bool,
    expected_version: int,
    resolved_by: str | None,
) -> tuple[TaskRecord, tuple[str, int] | None]:
    if task.version != expected_version:
        raise StaleTaskVersion(
            f"expected task version {expected_version}, found {task.version}"
        )
    approval = task.approval
    if (
        task.status is not TaskStatus.WAITING_APPROVAL
        or approval is None
        or approval.status is not ApprovalStatus.PENDING
        or approval.approval_id != approval_id
    ):
        raise InvalidTaskTransition("approval is stale or is not pending")
    if task.current_node is None:
        raise InvalidTaskTransition("approval task has no current node")
    node = task.nodes[task.current_node]
    if node.status is not NodeStatus.WAITING_APPROVAL:
        raise InvalidTaskTransition("current node is not waiting for approval")
    now = utc_now_iso()
    resolved = replace(
        approval,
        status=ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED,
        resolved_at=now,
        resolved_by=resolved_by,
    )
    nodes = dict(task.nodes)
    if approved:
        validate_task_transition(task.status, TaskStatus.QUEUED)
        validate_node_transition(node.status, NodeStatus.QUEUED)
        node = replace(
            node,
            status=NodeStatus.QUEUED,
            attempt=node.attempt + 1,
            started_at=None,
            finished_at=None,
        )
        target = TaskStatus.QUEUED
        queue_item: tuple[str, int] | None = (node.node_id, node.attempt)
        failure_class = task.failure_class
        error = task.error
        finished_at = None
    else:
        validate_task_transition(task.status, TaskStatus.FAILED)
        validate_node_transition(node.status, NodeStatus.FAILED)
        node = replace(
            node,
            status=NodeStatus.FAILED,
            finished_at=now,
            failure_class=node.failure_class or "approval_rejected",
            error=node.error or "approval rejected",
        )
        target = TaskStatus.FAILED
        queue_item = None
        failure_class = node.failure_class
        error = node.error
        finished_at = now
    nodes[node.node_id] = node
    return (
        replace(
            task,
            nodes=nodes,
            status=target,
            approval=resolved,
            failure_class=failure_class,
            error=error,
            finished_at=finished_at,
            updated_at=now,
            version=task.version + 1,
        ),
        queue_item,
    )


def _retry_failed_record(
    task: TaskRecord, expected_version: int
) -> tuple[TaskRecord, tuple[str, int]]:
    if task.version != expected_version:
        raise StaleTaskVersion(
            f"expected task version {expected_version}, found {task.version}"
        )
    if task.status is not TaskStatus.FAILED or task.current_node is None:
        raise InvalidTaskTransition("only a failed workflow node can be retried")
    node = task.nodes[task.current_node]
    if node.status is not NodeStatus.FAILED or node.attempt >= node.max_attempts:
        raise InvalidTaskTransition("failed node has no retry attempts remaining")
    queued = replace(
        node,
        status=NodeStatus.QUEUED,
        attempt=node.attempt + 1,
        started_at=None,
        finished_at=None,
        retry_at=None,
        failure_class=None,
        error=None,
    )
    # This is an explicit recovery operation, not an ordinary terminal transition.
    updated = _replace_node(
        task,
        queued,
        status=TaskStatus.QUEUED,
        finished_at=None,
        failure_class=None,
        error=None,
        approval=None,
    )
    return updated, (queued.node_id, queued.attempt)


def _event_number(event_id: str) -> tuple[int, int]:
    try:
        major, minor = event_id.split("-", 1)
        return int(major), int(minor)
    except (ValueError, AttributeError):
        return 0, 0


class InMemoryTaskStore:
    """Thread-safe test/development store with Redis-like stream semantics."""

    def __init__(self) -> None:
        self._tasks: dict[str, TaskRecord] = {}
        self._events: dict[str, list[TaskEvent]] = defaultdict(list)
        self._queue: deque[QueueMessage] = deque()
        self._cancelled: set[str] = set()
        self._next_message_id = 1
        self._condition = threading.Condition(threading.RLock())

    def create(self, task: TaskRecord) -> TaskRecord:
        with self._condition:
            if task.task_id in self._tasks:
                raise ValueError(f"task already exists: {task.task_id}")
            self._tasks[task.task_id] = task
            return task

    def submit(self, task: TaskRecord) -> str:
        with self._condition:
            self.create(task)
            if task.is_workflow:
                self.publish(
                    task.task_id,
                    "workflow.created",
                    {
                        "status": task.status.value,
                        "workflow_type": task.workflow_type,
                        "workflow_version": task.workflow_version,
                    },
                )
                node = task.nodes[task.current_node or ""]
                message_id = self.enqueue(
                    task.task_id,
                    node.node_id,
                    task.version,
                    node.attempt,
                )
                self.publish(
                    task.task_id,
                    "node.queued",
                    {"node_id": node.node_id, "attempt": node.attempt},
                )
            else:
                self.publish(task.task_id, "task.created", {"status": task.status.value})
                message_id = self.enqueue(task.task_id)
                self.publish(task.task_id, "task.queued", {"status": task.status.value})
            return message_id

    def get(self, task_id: str) -> TaskRecord | None:
        with self._condition:
            return self._tasks.get(task_id)

    def _require(self, task_id: str) -> TaskRecord:
        task = self.get(task_id)
        if task is None:
            raise TaskNotFoundError(task_id)
        return task

    def transition(self, task_id: str, target: TaskStatus, **changes: Any) -> TaskRecord:
        with self._condition:
            updated = _transition_record(self._require(task_id), target, changes)
            self._tasks[task_id] = updated
            self._condition.notify_all()
            return updated

    def update(self, task_id: str, **changes: Any) -> TaskRecord:
        with self._condition:
            task = self._require(task_id)
            updated = replace(
                task, updated_at=utc_now_iso(), version=task.version + 1, **changes
            )
            self._tasks[task_id] = updated
            self._condition.notify_all()
            return updated

    def enqueue(
        self,
        task_id: str,
        node_id: str | None = None,
        expected_version: int | None = None,
        attempt: int | None = None,
    ) -> str:
        with self._condition:
            message_id = f"{self._next_message_id}-0"
            self._next_message_id += 1
            self._queue.append(
                QueueMessage(
                    message_id=message_id,
                    task_id=task_id,
                    node_id=node_id,
                    expected_version=expected_version,
                    attempt=attempt,
                )
            )
            self._condition.notify_all()
            return message_id

    def claim(
        self, consumer: str, block_ms: int = 1000
    ) -> QueueMessage | None:
        del consumer
        deadline = time.monotonic() + max(0, block_ms) / 1000
        with self._condition:
            while not self._queue:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            return self._queue.popleft()

    def touch_claim(self, message_id: str, consumer: str) -> None:
        del message_id, consumer

    def heartbeat_node(
        self, task_id: str, message_id: str, consumer: str
    ) -> None:
        del message_id
        with self._condition:
            task = self._require(task_id)
            if task.status is TaskStatus.RUNNING and task.worker_id == consumer:
                self._tasks[task_id] = replace(
                    task, worker_heartbeat_at=utc_now_iso()
                )

    def ack(self, message_id: str) -> None:
        del message_id

    def request_cancel(self, task_id: str) -> TaskRecord:
        with self._condition:
            task = self._require(task_id)
            self._cancelled.add(task_id)
            updated = _request_cancel_record(task)
            self._tasks[task_id] = updated
            self._condition.notify_all()
            return updated

    def cancel_requested(self, task_id: str) -> bool:
        with self._condition:
            return task_id in self._cancelled

    def mark_node_running(
        self,
        task_id: str,
        node_id: str,
        *,
        expected_version: int,
        attempt: int,
        worker_id: str | None = None,
    ) -> TaskRecord:
        with self._condition:
            updated = _mark_node_running_record(
                self._require(task_id), node_id, expected_version, attempt, worker_id
            )
            self._tasks[task_id] = updated
            self.publish(
                task_id,
                "node.started",
                {"node_id": node_id, "attempt": attempt, "worker_id": worker_id},
            )
            return updated

    def commit_node_result(
        self,
        task_id: str,
        node_id: str,
        result: NodeResult,
        next_node: str | None,
        expected_version: int,
        approval: ApprovalRecord | None = None,
    ) -> TaskRecord:
        with self._condition:
            updated, queue_item = _commit_node_result_record(
                self._require(task_id),
                node_id,
                result,
                next_node,
                expected_version,
                approval,
            )
            self._tasks[task_id] = updated
            self.publish(
                task_id,
                "node.succeeded",
                {
                    "node_id": node_id,
                    "attempt": updated.nodes[node_id].attempt,
                    "result_refs": list(updated.nodes[node_id].result_refs),
                },
            )
            if approval is not None:
                self.publish(task_id, "approval.requested", approval.to_dict())
            elif queue_item is not None:
                queued_node, attempt = queue_item
                self.enqueue(task_id, queued_node, updated.version, attempt)
                self.publish(
                    task_id,
                    "node.queued",
                    {"node_id": queued_node, "attempt": attempt},
                )
            else:
                self.publish(task_id, "workflow.succeeded", {"status": "succeeded"})
            return updated

    def mark_node_waiting_external(
        self,
        task_id: str,
        node_id: str,
        result: NodeResult,
        *,
        expected_version: int,
    ) -> TaskRecord:
        with self._condition:
            updated = _waiting_external_record(
                self._require(task_id), node_id, result, expected_version
            )
            self._tasks[task_id] = updated
            self.publish(
                task_id,
                "training.submitted",
                {
                    "node_id": node_id,
                    "attempt": updated.nodes[node_id].attempt,
                    "external_job_id": result.external_job_id,
                },
            )
            return updated

    def fail_node(
        self,
        task_id: str,
        node_id: str,
        result: NodeResult,
        *,
        expected_version: int,
    ) -> TaskRecord:
        with self._condition:
            updated = _fail_node_record(
                self._require(task_id), node_id, result, expected_version
            )
            self._tasks[task_id] = updated
            data = {
                "node_id": node_id,
                "attempt": updated.nodes[node_id].attempt,
                "failure_class": result.failure_class,
                "error": result.error,
            }
            self.publish(task_id, "node.failed", data)
            self.publish(task_id, "workflow.failed", data)
            return updated

    def schedule_retry(
        self,
        task_id: str,
        node_id: str,
        result: NodeResult,
        *,
        expected_version: int,
    ) -> TaskRecord:
        with self._condition:
            updated, queue_item = _queue_retry_record(
                self._require(task_id), node_id, result, expected_version
            )
            self._tasks[task_id] = updated
            if not queue_item[0]:
                data = {
                    "node_id": node_id,
                    "attempt": updated.nodes[node_id].attempt,
                    "failure_class": result.failure_class,
                    "error": result.error,
                }
                self.publish(task_id, "node.failed", data)
                self.publish(task_id, "workflow.failed", data)
                return updated
            queued_node, attempt = queue_item
            self.publish(
                task_id,
                "node.retry_scheduled",
                {
                    "node_id": queued_node,
                    "attempt": attempt,
                    "failure_class": result.failure_class,
                    "retry_at": updated.nodes[queued_node].retry_at,
                },
            )
            self.enqueue(task_id, queued_node, updated.version, attempt)
            self.publish(
                task_id,
                "node.queued",
                {"node_id": queued_node, "attempt": attempt},
            )
            return updated

    def request_node_approval(
        self,
        task_id: str,
        node_id: str,
        approval: ApprovalRecord,
        *,
        expected_version: int,
        result: NodeResult | None = None,
    ) -> TaskRecord:
        with self._condition:
            updated = _request_node_approval_record(
                self._require(task_id), node_id, approval, expected_version, result
            )
            self._tasks[task_id] = updated
            self.publish(task_id, "approval.requested", approval.to_dict())
            return updated

    def resolve_approval(
        self,
        task_id: str,
        approval_id: str,
        *,
        approved: bool,
        expected_version: int,
        resolved_by: str | None = None,
    ) -> TaskRecord:
        with self._condition:
            updated, queue_item = _resolve_approval_record(
                self._require(task_id),
                approval_id,
                approved,
                expected_version,
                resolved_by,
            )
            self._tasks[task_id] = updated
            assert updated.approval is not None
            self.publish(
                task_id,
                "approval.resolved",
                {
                    "approval_id": approval_id,
                    "status": updated.approval.status.value,
                    "resolved_by": resolved_by,
                },
            )
            if queue_item is not None:
                node_id, attempt = queue_item
                self.enqueue(task_id, node_id, updated.version, attempt)
                self.publish(
                    task_id, "node.queued", {"node_id": node_id, "attempt": attempt}
                )
            else:
                self.publish(
                    task_id,
                    "workflow.failed",
                    {"failure_class": updated.failure_class, "error": updated.error},
                )
            return updated

    def retry_failed_node(self, task_id: str, *, expected_version: int) -> TaskRecord:
        with self._condition:
            updated, queue_item = _retry_failed_record(
                self._require(task_id), expected_version
            )
            self._tasks[task_id] = updated
            node_id, attempt = queue_item
            self.enqueue(task_id, node_id, updated.version, attempt)
            self.publish(
                task_id, "node.queued", {"node_id": node_id, "attempt": attempt}
            )
            return updated

    def waiting_external(self) -> list[TaskRecord]:
        with self._condition:
            return [
                task
                for task in self._tasks.values()
                if task.status in {
                    TaskStatus.WAITING_EXTERNAL,
                    TaskStatus.CANCELLING,
                }
                and task.current_node is not None
                and task.nodes[task.current_node].external_job_id is not None
            ]

    def publish(self, task_id: str, event_type: str, data: dict[str, Any]) -> TaskEvent:
        with self._condition:
            event_id = f"{self._next_message_id}-0"
            self._next_message_id += 1
            event = TaskEvent(event_id, task_id, event_type, utc_now_iso(), dict(data))
            self._events[task_id].append(event)
            self._condition.notify_all()
            return event

    def read_events(
        self, task_id: str, after_id: str = "0-0", block_ms: int = 0, count: int = 100
    ) -> list[TaskEvent]:
        deadline = time.monotonic() + max(0, block_ms) / 1000
        with self._condition:
            while True:
                events = [
                    event
                    for event in self._events.get(task_id, ())
                    if _event_number(event.event_id) > _event_number(after_id)
                ][:count]
                if events or block_ms <= 0:
                    return events
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return []
                self._condition.wait(remaining)

    def close(self) -> None:
        return None

    def ping(self) -> bool:
        return True


class RedisTaskStore:
    """Redis-backed state and streams with optimistic-lock state transitions."""

    def __init__(
        self,
        redis_url: str,
        *,
        namespace: str = "agent4ml:ml",
        max_events: int = 5000,
        reclaim_idle_ms: int = 60000,
    ) -> None:
        try:
            import redis
        except ImportError as exc:  # pragma: no cover - dependency error is explicit
            raise RuntimeError("install the 'redis' package to use RedisTaskStore") from exc
        self._redis_module = redis
        self._client = redis.Redis.from_url(redis_url, decode_responses=True)
        self._namespace = namespace.rstrip(":")
        self._max_events = max_events
        self._reclaim_idle_ms = reclaim_idle_ms
        self._queue_key = f"{self._namespace}:queue"
        self._group = f"{self._namespace}:workers"
        try:
            self._client.xgroup_create(self._queue_key, self._group, id="0", mkstream=True)
        except redis.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    def _task_key(self, task_id: str) -> str:
        return f"{self._namespace}:task:{task_id}"

    def _cancel_key(self, task_id: str) -> str:
        return f"{self._namespace}:cancel:{task_id}"

    def _events_key(self, task_id: str) -> str:
        return f"{self._namespace}:events:{task_id}"

    @staticmethod
    def _encode(task: TaskRecord) -> str:
        return json.dumps(task.to_dict(), ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _decode(raw: str | bytes) -> TaskRecord:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return TaskRecord.from_dict(json.loads(raw))

    def create(self, task: TaskRecord) -> TaskRecord:
        if not self._client.set(self._task_key(task.task_id), self._encode(task), nx=True):
            raise ValueError(f"task already exists: {task.task_id}")
        return task

    def _event_fields(
        self, task_id: str, event_type: str, data: dict[str, Any], timestamp: str
    ) -> dict[str, str]:
        return {
            "task_id": task_id,
            "event_type": event_type,
            "timestamp": timestamp,
            "data": json.dumps(data, ensure_ascii=False, separators=(",", ":")),
        }

    def submit(self, task: TaskRecord) -> str:
        task_key = self._task_key(task.task_id)
        events_key = self._events_key(task.task_id)
        redis = self._redis_module
        timestamp = utc_now_iso()
        with self._client.pipeline() as pipe:
            try:
                pipe.watch(task_key)
                if pipe.exists(task_key):
                    raise ValueError(f"task already exists: {task.task_id}")
                pipe.multi()
                pipe.set(task_key, self._encode(task))
                if task.is_workflow:
                    node = task.nodes[task.current_node or ""]
                    pipe.xadd(
                        events_key,
                        self._event_fields(
                            task.task_id,
                            "workflow.created",
                            {
                                "status": task.status.value,
                                "workflow_type": task.workflow_type,
                                "workflow_version": task.workflow_version,
                            },
                            timestamp,
                        ),
                        maxlen=self._max_events,
                        approximate=True,
                    )
                    pipe.xadd(
                        self._queue_key,
                        self._queue_fields(
                            task.task_id, node.node_id, task.version, node.attempt
                        ),
                    )
                    pipe.xadd(
                        events_key,
                        self._event_fields(
                            task.task_id,
                            "node.queued",
                            {"node_id": node.node_id, "attempt": node.attempt},
                            timestamp,
                        ),
                        maxlen=self._max_events,
                        approximate=True,
                    )
                else:
                    pipe.xadd(
                        events_key,
                        self._event_fields(
                            task.task_id,
                            "task.created",
                            {"status": task.status.value},
                            timestamp,
                        ),
                        maxlen=self._max_events,
                        approximate=True,
                    )
                    pipe.xadd(self._queue_key, {"task_id": task.task_id})
                    pipe.xadd(
                        events_key,
                        self._event_fields(
                            task.task_id,
                            "task.queued",
                            {"status": task.status.value},
                            timestamp,
                        ),
                        maxlen=self._max_events,
                        approximate=True,
                    )
                results = pipe.execute()
                return str(results[2])
            except redis.WatchError as exc:
                raise RuntimeError(f"concurrent task submission: {task.task_id}") from exc

    def get(self, task_id: str) -> TaskRecord | None:
        raw = self._client.get(self._task_key(task_id))
        return self._decode(raw) if raw else None

    def _mutate(self, task_id: str, callback: Any) -> TaskRecord:
        key = self._task_key(task_id)
        redis = self._redis_module
        for _ in range(8):
            with self._client.pipeline() as pipe:
                try:
                    pipe.watch(key)
                    raw = pipe.get(key)
                    if raw is None:
                        raise TaskNotFoundError(task_id)
                    updated = callback(self._decode(raw))
                    pipe.multi()
                    pipe.set(key, self._encode(updated))
                    pipe.execute()
                    return updated
                except redis.WatchError:
                    continue
        raise RuntimeError(f"concurrent task update did not converge: {task_id}")

    def transition(self, task_id: str, target: TaskStatus, **changes: Any) -> TaskRecord:
        return self._mutate(
            task_id, lambda task: _transition_record(task, target, changes)
        )

    def update(self, task_id: str, **changes: Any) -> TaskRecord:
        return self._mutate(
            task_id,
            lambda task: replace(
                task, updated_at=utc_now_iso(), version=task.version + 1, **changes
            ),
        )

    @staticmethod
    def _queue_fields(
        task_id: str,
        node_id: str | None = None,
        expected_version: int | None = None,
        attempt: int | None = None,
    ) -> dict[str, str]:
        fields = {"task_id": task_id}
        if node_id is not None:
            fields["node_id"] = node_id
        if expected_version is not None:
            fields["expected_version"] = str(expected_version)
        if attempt is not None:
            fields["attempt"] = str(attempt)
        return fields

    def enqueue(
        self,
        task_id: str,
        node_id: str | None = None,
        expected_version: int | None = None,
        attempt: int | None = None,
    ) -> str:
        return str(
            self._client.xadd(
                self._queue_key,
                self._queue_fields(task_id, node_id, expected_version, attempt),
            )
        )

    def claim(
        self, consumer: str, block_ms: int = 1000
    ) -> QueueMessage | None:
        recovered = self._client.xautoclaim(
            self._queue_key,
            self._group,
            consumer,
            min_idle_time=self._reclaim_idle_ms,
            start_id="0-0",
            count=1,
        )
        recovered_entries = recovered[1] if len(recovered) > 1 else []
        if recovered_entries:
            message_id, fields = recovered_entries[0]
            return QueueMessage(
                message_id=str(message_id),
                task_id=str(fields["task_id"]),
                recovered=True,
                node_id=fields.get("node_id"),
                expected_version=int(fields["expected_version"])
                if fields.get("expected_version") is not None
                else None,
                attempt=int(fields["attempt"])
                if fields.get("attempt") is not None
                else None,
            )
        messages = self._client.xreadgroup(
            self._group,
            consumer,
            {self._queue_key: ">"},
            count=1,
            block=max(1, block_ms),
        )
        if not messages:
            return None
        _, entries = messages[0]
        message_id, fields = entries[0]
        return QueueMessage(
            message_id=str(message_id),
            task_id=str(fields["task_id"]),
            node_id=fields.get("node_id"),
            expected_version=int(fields["expected_version"])
            if fields.get("expected_version") is not None
            else None,
            attempt=int(fields["attempt"])
            if fields.get("attempt") is not None
            else None,
        )

    def touch_claim(self, message_id: str, consumer: str) -> None:
        self._client.xclaim(
            self._queue_key,
            self._group,
            consumer,
            min_idle_time=0,
            message_ids=[message_id],
            justid=True,
        )

    def heartbeat_node(
        self, task_id: str, message_id: str, consumer: str
    ) -> None:
        def heartbeat(task: TaskRecord) -> TaskRecord:
            if task.status is TaskStatus.RUNNING and task.worker_id == consumer:
                return replace(task, worker_heartbeat_at=utc_now_iso())
            return task

        self._mutate(task_id, heartbeat)
        self.touch_claim(message_id, consumer)

    def ack(self, message_id: str) -> None:
        self._client.xack(self._queue_key, self._group, message_id)

    def request_cancel(self, task_id: str) -> TaskRecord:
        self._client.set(self._cancel_key(task_id), "1", ex=86400)

        def mutate(task: TaskRecord) -> TaskRecord:
            return _request_cancel_record(task)

        return self._mutate(task_id, mutate)

    def cancel_requested(self, task_id: str) -> bool:
        return bool(self._client.exists(self._cancel_key(task_id)))

    def _workflow_mutate(
        self,
        task_id: str,
        callback: Any,
    ) -> TaskRecord:
        """Atomically persist workflow state, events, and an optional next message."""
        key = self._task_key(task_id)
        events_key = self._events_key(task_id)
        redis = self._redis_module
        for _ in range(8):
            with self._client.pipeline() as pipe:
                try:
                    pipe.watch(key)
                    raw = pipe.get(key)
                    if raw is None:
                        raise TaskNotFoundError(task_id)
                    updated, queue_item, events = callback(self._decode(raw))
                    timestamp = utc_now_iso()
                    pipe.multi()
                    pipe.set(key, self._encode(updated))
                    for event_type, data in events:
                        pipe.xadd(
                            events_key,
                            self._event_fields(
                                task_id, event_type, data, timestamp
                            ),
                            maxlen=self._max_events,
                            approximate=True,
                        )
                    if queue_item is not None:
                        node_id, attempt = queue_item
                        pipe.xadd(
                            self._queue_key,
                            self._queue_fields(
                                task_id, node_id, updated.version, attempt
                            ),
                        )
                    pipe.execute()
                    return updated
                except redis.WatchError:
                    continue
        raise RuntimeError(f"concurrent workflow update did not converge: {task_id}")

    def mark_node_running(
        self,
        task_id: str,
        node_id: str,
        *,
        expected_version: int,
        attempt: int,
        worker_id: str | None = None,
    ) -> TaskRecord:
        def mutate(task: TaskRecord):
            updated = _mark_node_running_record(
                task, node_id, expected_version, attempt, worker_id
            )
            return updated, None, [
                (
                    "node.started",
                    {"node_id": node_id, "attempt": attempt, "worker_id": worker_id},
                )
            ]

        return self._workflow_mutate(task_id, mutate)

    def commit_node_result(
        self,
        task_id: str,
        node_id: str,
        result: NodeResult,
        next_node: str | None,
        expected_version: int,
        approval: ApprovalRecord | None = None,
    ) -> TaskRecord:
        def mutate(task: TaskRecord):
            updated, queue_item = _commit_node_result_record(
                task,
                node_id,
                result,
                next_node,
                expected_version,
                approval,
            )
            events: list[tuple[str, dict[str, Any]]] = [
                (
                    "node.succeeded",
                    {
                        "node_id": node_id,
                        "attempt": updated.nodes[node_id].attempt,
                        "result_refs": list(updated.nodes[node_id].result_refs),
                    },
                )
            ]
            if approval is not None:
                events.append(("approval.requested", approval.to_dict()))
            elif queue_item is not None:
                queued_node, attempt = queue_item
                events.append(
                    ("node.queued", {"node_id": queued_node, "attempt": attempt})
                )
            else:
                events.append(("workflow.succeeded", {"status": "succeeded"}))
            return updated, queue_item, events

        return self._workflow_mutate(task_id, mutate)

    def mark_node_waiting_external(
        self,
        task_id: str,
        node_id: str,
        result: NodeResult,
        *,
        expected_version: int,
    ) -> TaskRecord:
        def mutate(task: TaskRecord):
            updated = _waiting_external_record(
                task, node_id, result, expected_version
            )
            return updated, None, [
                (
                    "training.submitted",
                    {
                        "node_id": node_id,
                        "attempt": updated.nodes[node_id].attempt,
                        "external_job_id": result.external_job_id,
                    },
                )
            ]

        return self._workflow_mutate(task_id, mutate)

    def fail_node(
        self,
        task_id: str,
        node_id: str,
        result: NodeResult,
        *,
        expected_version: int,
    ) -> TaskRecord:
        def mutate(task: TaskRecord):
            updated = _fail_node_record(task, node_id, result, expected_version)
            data = {
                "node_id": node_id,
                "attempt": updated.nodes[node_id].attempt,
                "failure_class": result.failure_class,
                "error": result.error,
            }
            return updated, None, [("node.failed", data), ("workflow.failed", data)]

        return self._workflow_mutate(task_id, mutate)

    def schedule_retry(
        self,
        task_id: str,
        node_id: str,
        result: NodeResult,
        *,
        expected_version: int,
    ) -> TaskRecord:
        def mutate(task: TaskRecord):
            updated, queue_item = _queue_retry_record(
                task, node_id, result, expected_version
            )
            if not queue_item[0]:
                data = {
                    "node_id": node_id,
                    "attempt": updated.nodes[node_id].attempt,
                    "failure_class": result.failure_class,
                    "error": result.error,
                }
                return updated, None, [
                    ("node.failed", data),
                    ("workflow.failed", data),
                ]
            queued_node, attempt = queue_item
            return updated, queue_item, [
                (
                    "node.retry_scheduled",
                    {
                        "node_id": queued_node,
                        "attempt": attempt,
                        "failure_class": result.failure_class,
                        "retry_at": updated.nodes[queued_node].retry_at,
                    },
                ),
                ("node.queued", {"node_id": queued_node, "attempt": attempt}),
            ]

        return self._workflow_mutate(task_id, mutate)

    def request_node_approval(
        self,
        task_id: str,
        node_id: str,
        approval: ApprovalRecord,
        *,
        expected_version: int,
        result: NodeResult | None = None,
    ) -> TaskRecord:
        def mutate(task: TaskRecord):
            updated = _request_node_approval_record(
                task, node_id, approval, expected_version, result
            )
            return updated, None, [("approval.requested", approval.to_dict())]

        return self._workflow_mutate(task_id, mutate)

    def resolve_approval(
        self,
        task_id: str,
        approval_id: str,
        *,
        approved: bool,
        expected_version: int,
        resolved_by: str | None = None,
    ) -> TaskRecord:
        def mutate(task: TaskRecord):
            updated, queue_item = _resolve_approval_record(
                task, approval_id, approved, expected_version, resolved_by
            )
            assert updated.approval is not None
            events: list[tuple[str, dict[str, Any]]] = [
                (
                    "approval.resolved",
                    {
                        "approval_id": approval_id,
                        "status": updated.approval.status.value,
                        "resolved_by": resolved_by,
                    },
                )
            ]
            if queue_item is not None:
                queued_node, attempt = queue_item
                events.append(
                    ("node.queued", {"node_id": queued_node, "attempt": attempt})
                )
            else:
                events.append(
                    (
                        "workflow.failed",
                        {
                            "failure_class": updated.failure_class,
                            "error": updated.error,
                        },
                    )
                )
            return updated, queue_item, events

        return self._workflow_mutate(task_id, mutate)

    def retry_failed_node(self, task_id: str, *, expected_version: int) -> TaskRecord:
        def mutate(task: TaskRecord):
            updated, queue_item = _retry_failed_record(task, expected_version)
            node_id, attempt = queue_item
            return updated, queue_item, [
                ("node.queued", {"node_id": node_id, "attempt": attempt})
            ]

        return self._workflow_mutate(task_id, mutate)

    def waiting_external(self) -> list[TaskRecord]:
        tasks: list[TaskRecord] = []
        pattern = f"{self._namespace}:task:*"
        for key in self._client.scan_iter(match=pattern, count=100):
            raw = self._client.get(key)
            if not raw:
                continue
            task = self._decode(raw)
            if (
                task.status in {TaskStatus.WAITING_EXTERNAL, TaskStatus.CANCELLING}
                and task.current_node is not None
                and task.nodes[task.current_node].external_job_id is not None
            ):
                tasks.append(task)
        return tasks

    def publish(self, task_id: str, event_type: str, data: dict[str, Any]) -> TaskEvent:
        timestamp = utc_now_iso()
        event_id = self._client.xadd(
            self._events_key(task_id),
            self._event_fields(task_id, event_type, data, timestamp),
            maxlen=self._max_events,
            approximate=True,
        )
        return TaskEvent(str(event_id), task_id, event_type, timestamp, dict(data))

    def read_events(
        self, task_id: str, after_id: str = "0-0", block_ms: int = 0, count: int = 100
    ) -> list[TaskEvent]:
        streams = self._client.xread(
            {self._events_key(task_id): after_id},
            count=count,
            block=block_ms or None,
        )
        if not streams:
            return []
        _, entries = streams[0]
        return [
            TaskEvent(
                event_id=str(event_id),
                task_id=str(fields["task_id"]),
                event_type=str(fields["event_type"]),
                timestamp=str(fields["timestamp"]),
                data=json.loads(fields.get("data", "{}")),
            )
            for event_id, fields in entries
        ]

    def close(self) -> None:
        self._client.close()

    def ping(self) -> bool:
        return bool(self._client.ping())
