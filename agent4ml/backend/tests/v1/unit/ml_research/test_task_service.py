from __future__ import annotations

import pytest

from agent4ml.backend.agents.ml_research.task_models import (
    InvalidTaskTransition,
    TaskStatus,
)
from agent4ml.backend.agents.ml_research.task_service import MLTaskService, TaskValidationError
from agent4ml.backend.agents.ml_research.task_store import InMemoryTaskStore


def _service(tmp_path):
    store = InMemoryTaskStore()
    return store, MLTaskService(store, work_root=tmp_path / "runs", allowed_root=tmp_path)


def test_submit_persists_and_enqueues_task_with_replayable_events(tmp_path):
    store, service = _service(tmp_path)

    task = service.submit(name="demo", command=["python", "train.py"], cwd=tmp_path)
    claimed = store.claim("worker-1", block_ms=0)
    events = store.read_events(task.task_id, "0-0")

    assert task.status is TaskStatus.QUEUED
    assert claimed is not None and claimed[1] == task.task_id
    assert claimed[2] is False
    assert [event.event_type for event in events] == ["task.created", "task.queued"]
    replay = store.read_events(task.task_id, events[0].event_id)
    assert [event.event_type for event in replay] == ["task.queued"]


def test_cancel_queued_task_is_immediately_terminal(tmp_path):
    store, service = _service(tmp_path)
    task = service.submit(name="demo", command=["python", "train.py"], cwd=tmp_path)

    cancelled = service.cancel(task.task_id)

    assert cancelled.status is TaskStatus.CANCELLED
    assert store.cancel_requested(task.task_id)
    assert store.read_events(task.task_id)[-1].event_type == "task.cancelled"


def test_cancel_running_task_enters_cancelling(tmp_path):
    store, service = _service(tmp_path)
    task = service.submit(name="demo", command=["python", "train.py"], cwd=tmp_path)
    store.transition(task.task_id, TaskStatus.RUNNING)

    cancelling = service.cancel(task.task_id)

    assert cancelling.status is TaskStatus.CANCELLING


def test_state_machine_rejects_success_after_cancel(tmp_path):
    store, service = _service(tmp_path)
    task = service.submit(name="demo", command=["python", "train.py"], cwd=tmp_path)
    service.cancel(task.task_id)

    with pytest.raises(InvalidTaskTransition):
        store.transition(task.task_id, TaskStatus.SUCCEEDED)


def test_submit_rejects_working_directory_outside_allowed_root(tmp_path):
    store = InMemoryTaskStore()
    service = MLTaskService(
        store, work_root=tmp_path / "runs", allowed_root=tmp_path / "allowed"
    )

    with pytest.raises(TaskValidationError, match="AGENT4ML_ML_ALLOWED_ROOT"):
        service.submit(name="demo", command=["python"], cwd=tmp_path)
