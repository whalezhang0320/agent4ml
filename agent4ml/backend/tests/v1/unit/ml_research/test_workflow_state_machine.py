from __future__ import annotations

import sys
import threading
import time

import pytest

from agent4ml.backend.agents.ml_research.task_models import (
    NodeStatus,
    StaleTaskVersion,
    TaskStatus,
)
from agent4ml.backend.agents.ml_research.task_service import MLTaskService
from agent4ml.backend.agents.ml_research.task_store import InMemoryTaskStore
from agent4ml.backend.agents.ml_research.task_worker import MLTaskWorker
from agent4ml.backend.agents.ml_research.training_reconciler import TrainingReconciler
from agent4ml.backend.agents.ml_research.workflow import NodeResult


def _setup(tmp_path, *, handlers=None):
    store = InMemoryTaskStore()
    service = MLTaskService(store, work_root=tmp_path / "runs", allowed_root=tmp_path)
    worker = MLTaskWorker(store, worker_id="workflow-worker", handlers=handlers)
    return store, service, worker


def test_reproduction_pauses_for_approval_and_resumes_from_training(tmp_path):
    store, service, worker = _setup(tmp_path)
    task = service.submit_reproduction(
        name="reproduce",
        repository_path=tmp_path,
        target_metric="accuracy",
        training_command=[sys.executable, "-c", "print('trained')"],
    )

    assert worker.run_once(block_ms=0)  # analyze_code
    assert worker.run_once(block_ms=0)  # build_environment
    waiting = service.get(task.task_id)

    assert waiting.status is TaskStatus.WAITING_APPROVAL
    assert waiting.current_node == "run_training"
    assert waiting.nodes["analyze_code"].status is NodeStatus.SUCCEEDED
    assert waiting.nodes["build_environment"].status is NodeStatus.SUCCEEDED
    assert waiting.nodes["run_training"].status is NodeStatus.WAITING_APPROVAL
    assert waiting.approval is not None

    approved = service.approve(
        task.task_id,
        approval_id=waiting.approval.approval_id,
        expected_version=waiting.version,
        resolved_by="tester",
    )
    assert approved.nodes["run_training"].attempt == 1
    assert worker.run_once(block_ms=0)  # run_training
    assert worker.run_once(block_ms=0)  # validate_result

    completed = service.get(task.task_id)
    assert completed.status is TaskStatus.SUCCEEDED
    assert completed.current_node is None
    assert set(completed.artifacts) == {
        "repository_analysis",
        "environment_manifest",
        "training_run",
        "validation_result",
    }
    event_types = [event.event_type for event in store.read_events(task.task_id)]
    assert "approval.requested" in event_types
    assert "approval.resolved" in event_types
    assert event_types[-1] == "workflow.succeeded"


def test_stale_queue_message_is_acked_without_duplicate_node_execution(tmp_path):
    class CountingHandler:
        def __init__(self):
            self.calls = 0

        def execute(self, task, node):
            del task, node
            self.calls += 1
            return NodeResult.succeeded()

    analyze = CountingHandler()
    handlers = {
        "analyze_code": analyze,
        "build_environment": CountingHandler(),
        "run_training": CountingHandler(),
        "validate_result": CountingHandler(),
    }
    store, service, worker = _setup(tmp_path, handlers=handlers)
    task = service.submit_reproduction(name="reproduce", repository_path=tmp_path)
    original = store.claim("manual", block_ms=0)
    assert original is not None
    store.enqueue(
        task.task_id,
        original.node_id,
        original.expected_version,
        original.attempt,
    )
    # Put the original back as an equivalent message and execute exactly one copy.
    store.enqueue(
        task.task_id,
        original.node_id,
        original.expected_version,
        original.attempt,
    )

    assert worker.run_once(block_ms=0)
    assert worker.run_once(block_ms=0)
    assert analyze.calls == 1


def test_approval_rejects_stale_task_version(tmp_path):
    _, service, worker = _setup(tmp_path)
    task = service.submit_reproduction(name="reproduce", repository_path=tmp_path)
    worker.run_once(block_ms=0)
    worker.run_once(block_ms=0)
    waiting = service.get(task.task_id)
    assert waiting.approval is not None

    with pytest.raises(StaleTaskVersion):
        service.approve(
            task.task_id,
            approval_id=waiting.approval.approval_id,
            expected_version=waiting.version - 1,
        )


def test_network_failure_retries_only_current_node(tmp_path):
    class FlakyAnalyze:
        def __init__(self):
            self.calls = 0

        def execute(self, task, node):
            del task, node
            self.calls += 1
            if self.calls == 1:
                return NodeResult.failed("network_timeout", "temporary timeout")
            return NodeResult.succeeded()

    flaky = FlakyAnalyze()
    handlers = {
        "analyze_code": flaky,
        "build_environment": flaky,
        "run_training": flaky,
        "validate_result": flaky,
    }
    store, service, worker = _setup(tmp_path, handlers=handlers)
    task = service.submit_reproduction(name="reproduce", repository_path=tmp_path)

    worker.run_once(block_ms=0)
    retrying = service.get(task.task_id)
    assert retrying.status is TaskStatus.QUEUED
    assert retrying.current_node == "analyze_code"
    assert retrying.nodes["analyze_code"].attempt == 2
    assert retrying.nodes["build_environment"].status is NodeStatus.PENDING

    worker.run_once(block_ms=0)
    advanced = service.get(task.task_id)
    assert advanced.nodes["analyze_code"].status is NodeStatus.SUCCEEDED
    assert advanced.current_node == "build_environment"
    assert "node.retry_scheduled" in [
        event.event_type for event in store.read_events(task.task_id)
    ]


def test_external_training_is_reconciled_without_duplicate_submission(tmp_path):
    class SuccessHandler:
        def execute(self, task, node):
            del task, node
            return NodeResult.succeeded()

    class Backend:
        def __init__(self):
            self.submissions = 0

        def submit(self, *, config_path, idempotency_key, inputs):
            del config_path, idempotency_key, inputs
            self.submissions += 1
            return "job-123"

        def get_status(self, external_job_id):
            assert external_job_id == "job-123"
            return "SUCCEEDED"

    class ExternalTrainingHandler:
        def __init__(self, backend):
            self.backend = backend

        def execute(self, task, node):
            job_id = self.backend.submit(
                config_path="",
                idempotency_key=f"{task.task_id}:{node.node_id}:{node.attempt}",
                inputs=task.inputs,
            )
            return NodeResult.waiting_external(job_id)

    backend = Backend()
    success = SuccessHandler()
    handlers = {
        "analyze_code": success,
        "build_environment": success,
        "run_training": ExternalTrainingHandler(backend),
        "validate_result": success,
    }
    store, service, worker = _setup(tmp_path, handlers=handlers)
    task = service.submit_reproduction(name="external", repository_path=tmp_path)
    worker.run_once(block_ms=0)
    worker.run_once(block_ms=0)
    waiting = service.get(task.task_id)
    assert waiting.approval is not None
    service.approve(
        task.task_id,
        approval_id=waiting.approval.approval_id,
        expected_version=waiting.version,
    )
    worker.run_once(block_ms=0)

    external = service.get(task.task_id)
    assert external.status is TaskStatus.WAITING_EXTERNAL
    assert external.nodes["run_training"].external_job_id == "job-123"
    assert backend.submissions == 1

    assert TrainingReconciler(store, backend).run_once() == 1
    reconciled = service.get(task.task_id)
    assert reconciled.status is TaskStatus.QUEUED
    assert reconciled.current_node == "validate_result"
    assert backend.submissions == 1


def test_running_workflow_training_can_be_cancelled(tmp_path):
    _, service, worker = _setup(tmp_path)
    task = service.submit_reproduction(
        name="cancel-training",
        repository_path=tmp_path,
        training_command=[sys.executable, "-c", "import time; time.sleep(30)"],
    )
    worker.run_once(block_ms=0)
    worker.run_once(block_ms=0)
    waiting = service.get(task.task_id)
    assert waiting.approval is not None
    service.approve(
        task.task_id,
        approval_id=waiting.approval.approval_id,
        expected_version=waiting.version,
    )
    thread = threading.Thread(target=worker.run_once, kwargs={"block_ms": 0})
    thread.start()
    deadline = time.monotonic() + 3
    while service.get(task.task_id).status is not TaskStatus.RUNNING:
        assert time.monotonic() < deadline
        time.sleep(0.01)

    service.cancel(task.task_id)
    thread.join(timeout=3)

    cancelled = service.get(task.task_id)
    assert not thread.is_alive()
    assert cancelled.status is TaskStatus.CANCELLED
    assert cancelled.nodes["run_training"].status is NodeStatus.SKIPPED
