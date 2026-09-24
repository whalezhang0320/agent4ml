"""Storage protocol plus in-memory and Redis implementations for ML tasks."""
from __future__ import annotations

import json
import threading
import time
from collections import defaultdict, deque
from dataclasses import replace
from typing import Any, Protocol

from agent4ml.backend.agents.ml_research.task_models import (
    ALLOWED_TRANSITIONS,
    InvalidTaskTransition,
    TaskEvent,
    TaskRecord,
    TaskStatus,
    utc_now_iso,
)


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
    def enqueue(self, task_id: str) -> str: ...
    def claim(
        self, consumer: str, block_ms: int = 1000
    ) -> tuple[str, str, bool] | None: ...
    def touch_claim(self, message_id: str, consumer: str) -> None: ...
    def ack(self, message_id: str) -> None: ...
    def request_cancel(self, task_id: str) -> TaskRecord: ...
    def cancel_requested(self, task_id: str) -> bool: ...
    def publish(self, task_id: str, event_type: str, data: dict[str, Any]) -> TaskEvent: ...
    def read_events(
        self, task_id: str, after_id: str = "0-0", block_ms: int = 0, count: int = 100
    ) -> list[TaskEvent]: ...
    def close(self) -> None: ...
    def ping(self) -> bool: ...


def _transition_record(
    task: TaskRecord, target: TaskStatus, changes: dict[str, Any]
) -> TaskRecord:
    if target not in ALLOWED_TRANSITIONS[task.status]:
        raise InvalidTaskTransition(f"cannot transition {task.status.value} -> {target.value}")
    now = utc_now_iso()
    defaults: dict[str, Any] = {"updated_at": now, "version": task.version + 1}
    if target is TaskStatus.RUNNING:
        defaults["started_at"] = task.started_at or now
    if target in {TaskStatus.CANCELLED, TaskStatus.SUCCEEDED, TaskStatus.FAILED}:
        defaults["finished_at"] = now
    defaults.update(changes)
    return replace(task, status=target, **defaults)


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
        self._queue: deque[tuple[str, str]] = deque()
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

    def enqueue(self, task_id: str) -> str:
        with self._condition:
            message_id = f"{self._next_message_id}-0"
            self._next_message_id += 1
            self._queue.append((message_id, task_id))
            self._condition.notify_all()
            return message_id

    def claim(
        self, consumer: str, block_ms: int = 1000
    ) -> tuple[str, str, bool] | None:
        del consumer
        deadline = time.monotonic() + max(0, block_ms) / 1000
        with self._condition:
            while not self._queue:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            message_id, task_id = self._queue.popleft()
            return message_id, task_id, False

    def touch_claim(self, message_id: str, consumer: str) -> None:
        del message_id, consumer

    def ack(self, message_id: str) -> None:
        del message_id

    def request_cancel(self, task_id: str) -> TaskRecord:
        with self._condition:
            task = self._require(task_id)
            self._cancelled.add(task_id)
            if task.status is TaskStatus.QUEUED:
                updated = _transition_record(task, TaskStatus.CANCELLED, {})
            elif task.status is TaskStatus.RUNNING:
                updated = _transition_record(task, TaskStatus.CANCELLING, {})
            else:
                updated = task
            self._tasks[task_id] = updated
            self._condition.notify_all()
            return updated

    def cancel_requested(self, task_id: str) -> bool:
        with self._condition:
            return task_id in self._cancelled

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
                pipe.xadd(
                    events_key,
                    self._event_fields(
                        task.task_id, "task.created", {"status": task.status.value}, timestamp
                    ),
                    maxlen=self._max_events,
                    approximate=True,
                )
                pipe.xadd(self._queue_key, {"task_id": task.task_id})
                pipe.xadd(
                    events_key,
                    self._event_fields(
                        task.task_id, "task.queued", {"status": task.status.value}, timestamp
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

    def enqueue(self, task_id: str) -> str:
        return str(self._client.xadd(self._queue_key, {"task_id": task_id}))

    def claim(
        self, consumer: str, block_ms: int = 1000
    ) -> tuple[str, str, bool] | None:
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
            return str(message_id), str(fields["task_id"]), True
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
        return str(message_id), str(fields["task_id"]), False

    def touch_claim(self, message_id: str, consumer: str) -> None:
        self._client.xclaim(
            self._queue_key,
            self._group,
            consumer,
            min_idle_time=0,
            message_ids=[message_id],
            justid=True,
        )

    def ack(self, message_id: str) -> None:
        self._client.xack(self._queue_key, self._group, message_id)

    def request_cancel(self, task_id: str) -> TaskRecord:
        self._client.set(self._cancel_key(task_id), "1", ex=86400)

        def mutate(task: TaskRecord) -> TaskRecord:
            if task.status is TaskStatus.QUEUED:
                return _transition_record(task, TaskStatus.CANCELLED, {})
            if task.status is TaskStatus.RUNNING:
                return _transition_record(task, TaskStatus.CANCELLING, {})
            return task

        return self._mutate(task_id, mutate)

    def cancel_requested(self, task_id: str) -> bool:
        return bool(self._client.exists(self._cancel_key(task_id)))

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
