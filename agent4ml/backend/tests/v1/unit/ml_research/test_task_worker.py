from __future__ import annotations

import sys
import threading
import time

from agent4ml.backend.agents.ml_research.task_models import TaskStatus
from agent4ml.backend.agents.ml_research.task_service import MLTaskService
from agent4ml.backend.agents.ml_research.task_store import InMemoryTaskStore
from agent4ml.backend.agents.ml_research.task_worker import MLTaskWorker


class _RecoveredClaimStore(InMemoryTaskStore):
    def claim(self, consumer: str, block_ms: int = 1000):
        claim = super().claim(consumer, block_ms)
        if claim is None:
            return None
        message_id, task_id, _ = claim
        return message_id, task_id, True


def _setup(tmp_path):
    store = InMemoryTaskStore()
    service = MLTaskService(store, work_root=tmp_path / "runs", allowed_root=tmp_path)
    worker = MLTaskWorker(
        store, worker_id="test-worker", heartbeat_interval=0.01, cancel_grace_seconds=0.2
    )
    return store, service, worker


def test_worker_runs_task_and_streams_metric(tmp_path):
    store, service, worker = _setup(tmp_path)
    task = service.submit(
        name="success",
        command=[
            sys.executable,
            "-c",
            'print(\'AGENT4ML_METRIC {"step": 1, "metrics": {"loss": 0.5}}\')',
        ],
        cwd=tmp_path,
    )

    assert worker.run_once(block_ms=0)

    completed = service.get(task.task_id)
    event_types = [event.event_type for event in store.read_events(task.task_id)]
    assert completed.status is TaskStatus.SUCCEEDED
    assert "metric.reported" in event_types
    assert "task.succeeded" in event_types


def test_worker_classifies_cuda_oom(tmp_path):
    store, service, worker = _setup(tmp_path)
    task = service.submit(
        name="oom",
        command=[
            sys.executable,
            "-c",
            "import sys; print('torch.cuda.OutOfMemoryError: CUDA out of memory'); sys.exit(1)",
        ],
        cwd=tmp_path,
    )

    worker.run_once(block_ms=0)

    failed = service.get(task.task_id)
    assert failed.status is TaskStatus.FAILED
    assert failed.failure_class == "cuda_oom"
    failure_events = [
        event for event in store.read_events(task.task_id)
        if event.event_type == "failure.detected"
    ]
    assert failure_events[0].data["failure_class"] == "cuda_oom"


def test_worker_classifies_dependency_conflict(tmp_path):
    store, service, worker = _setup(tmp_path)
    task = service.submit(
        name="deps",
        command=[
            sys.executable,
            "-c",
            "import sys; print('ERROR: ResolutionImpossible: conflicting dependencies'); sys.exit(1)",
        ],
        cwd=tmp_path,
    )

    worker.run_once(block_ms=0)

    assert service.get(task.task_id).failure_class == "dependency_conflict"


def test_running_task_can_be_cancelled(tmp_path):
    store, service, worker = _setup(tmp_path)
    task = service.submit(
        name="cancel",
        command=[sys.executable, "-c", "import time; print('started', flush=True); time.sleep(30)"],
        cwd=tmp_path,
    )
    thread = threading.Thread(target=worker.run_once, kwargs={"block_ms": 0})
    thread.start()

    deadline = time.monotonic() + 3
    while service.get(task.task_id).status is not TaskStatus.RUNNING:
        assert time.monotonic() < deadline
        time.sleep(0.01)
    service.cancel(task.task_id)
    thread.join(timeout=3)

    assert not thread.is_alive()
    assert service.get(task.task_id).status is TaskStatus.CANCELLED


def test_recovered_stale_running_claim_is_failed_without_duplicate_execution(tmp_path):
    store = _RecoveredClaimStore()
    service = MLTaskService(store, work_root=tmp_path / "runs", allowed_root=tmp_path)
    task = service.submit(
        name="stale",
        command=[sys.executable, "-c", "raise AssertionError('must not execute')"],
        cwd=tmp_path,
    )
    store.transition(task.task_id, TaskStatus.RUNNING, worker_id="dead-worker")
    worker = MLTaskWorker(store, worker_id="replacement-worker")

    worker.run_once(block_ms=0)

    failed = service.get(task.task_id)
    assert failed.status is TaskStatus.FAILED
    assert failed.failure_class == "worker_lost"


def test_tracker_initialization_failure_does_not_leave_task_queued(tmp_path, monkeypatch):
    store, service, worker = _setup(tmp_path)
    task = service.submit(
        name="unwritable",
        command=[sys.executable, "-c", "print('must not execute')"],
        cwd=tmp_path,
    )

    def fail_initialize(*args, **kwargs):
        del args, kwargs
        raise OSError("run directory is not writable")

    monkeypatch.setattr(
        "agent4ml.backend.agents.ml_research.task_worker.LocalExperimentTracker.initialize",
        fail_initialize,
    )

    worker.run_once(block_ms=0)

    failed = service.get(task.task_id)
    assert failed.status is TaskStatus.FAILED
    assert failed.failure_class == "worker_error"
